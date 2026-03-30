# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools  # 导入functools模块，用于缓存属性等功能
import struct  # 导入struct模块，用于字节与浮点数之间的转换
from dataclasses import dataclass  # 导入dataclass装饰器，用于创建数据类
from enum import Enum  # 导入Enum枚举基类

_SCALAR_TYPES_ID_MAP = {}  # 全局字典，用于存储标量类型ID到ScalarType实例的映射


# Mirrors enum in `core/scalar_type.hpp`
class NanRepr(Enum):
    """NaN（非数值）表示方式的枚举类，镜像了C++中core/scalar_type.hpp的枚举定义。"""
    NONE = 0  # nans are not supported  # 不支持NaN
    IEEE_754 = 1  # nans are: Exp all 1s, mantissa not all 0s  # IEEE 754标准NaN：指数全1，尾数不全为0
    EXTD_RANGE_MAX_MIN = 2  # nans are: Exp all 1s, mantissa all 1s  # 扩展范围NaN：指数全1，尾数全1


# This ScalarType class is a parallel implementation of the C++ ScalarType
# class found in csrc/core/scalar_type.hpp.  These two classes should be kept
# in sync until the inductor fully supports custom C++ classes.
@dataclass(frozen=True)  # 冻结的数据类，实例创建后不可修改
class ScalarType:
    """
    ScalarType can represent a wide range of floating point and integer
    types, in particular it can be used to represent sub-byte data types
    (something that torch.dtype currently does not support). It is also
    capable of  representing types with a bias, i.e.:
      `stored_value = value + bias`,
    this is useful for quantized types (e.g. standard GPTQ 4bit uses a bias
    of 8). The implementation for this class can be found in
    csrc/core/scalar_type.hpp, these type signatures should be kept in sync
    with that file.

    标量类型类，可以表示多种浮点和整数类型，特别是可以表示子字节数据类型
    （torch.dtype目前不支持的类型）。还支持带偏置的类型表示，即：
      `存储值 = 值 + 偏置`，
    这对量化类型很有用（例如标准GPTQ 4bit使用偏置8）。
    该类的C++实现在csrc/core/scalar_type.hpp中，两边的类型签名应保持同步。
    """

    exponent: int  # 指数位数（浮点类型），整数类型时为0
    """
    Number of bits in the exponent if this is a floating point type
    (zero if this an integer type)
    """

    mantissa: int  # 尾数位数（浮点类型），或整数位数（不含符号位）
    """
    Number of bits in the mantissa if this is a floating point type,
    or the number bits representing an integer excluding the sign bit if
    this an integer type.
    """

    signed: bool  # 是否为有符号类型（即是否有符号位）
    "If the type is signed (i.e. has a sign bit)"

    bias: int  # 编码偏置值（存储值 = 实际值 + 偏置，默认为0）
    """
    bias used to encode the values in this scalar type
    (value = stored_value - bias, default 0) for example if we store the
    type as an unsigned integer with a bias of 128 then the value 0 will be
    stored as 128 and -1 will be stored as 127 and 1 will be stored as 129.
    """

    _finite_values_only: bool = False  # 是否仅支持有限值（不支持无穷大），默认False
    """
    Private: if infs are supported, used `has_infs()` instead.
    """

    nan_repr: NanRepr = NanRepr.IEEE_754  # NaN的表示方式，默认为IEEE 754标准
    """
    How NaNs are represent in this scalar type, returns NanRepr value.
    (not applicable for integer types)
    """

    def _floating_point_max_int(self) -> int:
        """
        计算浮点类型最大值的整数位表示（以IEEE 754双精度格式存储）。

        返回:
            int: 双精度浮点数格式下的最大值的原始整数位表示
        """
        assert self.mantissa <= 52 and self.exponent <= 11, (  # 断言尾数和指数位数不超过双精度浮点数的限制
            f"Cannot represent max/min as a double for type {self.__str__()}"
        )

        max_mantissa = (1 << self.mantissa) - 1  # 计算最大尾数值（所有尾数位全为1）
        if self.nan_repr == NanRepr.EXTD_RANGE_MAX_MIN:  # 如果NaN用扩展范围表示
            max_mantissa = max_mantissa - 1  # 最大尾数减1（因为全1被保留给NaN）

        max_exponent = (1 << self.exponent) - 2  # 计算最大指数值（指数全1通常保留给特殊值）
        if self.nan_repr == NanRepr.EXTD_RANGE_MAX_MIN or self.nan_repr == NanRepr.NONE:  # 如果不需要保留指数全1给NaN/Inf
            assert self.exponent < 11, (  # 断言指数位数小于11（双精度限制）
                f"Cannot represent max/min as a double for type {self.__str__()}"
            )
            max_exponent = max_exponent + 1  # 最大指数加1（可以使用指数全1的值）

        # adjust the exponent to match that of a double
        # for now we assume the exponent bias is the standard 2^(e-1) -1, (where
        # e is the exponent bits), there is some precedent for non-standard
        # biases, example `float8_e4m3b11fnuz` here:
        # https://github.com/jax-ml/ml_dtypes but to avoid premature over
        # complication we are just assuming the standard exponent bias until
        # there is a need to support non-standard biases
        exponent_bias = (1 << (self.exponent - 1)) - 1  # 计算当前类型的指数偏置：2^(e-1) - 1
        exponent_bias_double = (1 << 10) - 1  # double e = 11  # 双精度浮点数的指数偏置：1023

        max_exponent_double = max_exponent - exponent_bias + exponent_bias_double  # 将最大指数转换为双精度浮点数的指数表示

        # shift the mantissa and exponent into the proper positions for an
        # IEEE double and bitwise-or them together.
        return (max_mantissa << (52 - self.mantissa)) | (max_exponent_double << 52)  # 将尾数和指数移位到双精度格式的正确位置并组合

    def _floating_point_max(self) -> float:
        """
        计算浮点类型的最大可表示值。

        返回:
            float: 该浮点类型能表示的最大值
        """
        double_raw = self._floating_point_max_int()  # 获取最大值的原始整数位表示
        return struct.unpack("!d", struct.pack("!Q", double_raw))[0]  # 将整数位表示转换为双精度浮点数

    def _raw_max(self) -> int | float:
        """
        计算原始最大值（不考虑偏置）。

        返回:
            int | float: 浮点类型返回float，整数类型返回int
        """
        if self.is_floating_point():  # 如果是浮点类型
            return self._floating_point_max()  # 返回浮点最大值
        else:  # 如果是整数类型
            assert self.size_bits < 64 or self.size_bits == 64 and self.is_signed(), (  # 断言位数在可表示范围内
                "Cannot represent max as an int"
            )
            return (1 << self.mantissa) - 1  # 返回整数最大值：2^mantissa - 1

    def _raw_min(self) -> int | float:
        """
        计算原始最小值（不考虑偏置）。

        返回:
            int | float: 浮点类型返回float，整数类型返回int
        """
        if self.is_floating_point():  # 如果是浮点类型
            assert self.is_signed(), (  # 断言浮点类型是有符号的
                "We currently assume all floating point types are signed"
            )
            sign_bit_double = 1 << 63  # 双精度浮点数的符号位位置（最高位）

            max_raw = self._floating_point_max_int()  # 获取最大值的原始整数位表示
            min_raw = max_raw | sign_bit_double  # 将符号位设为1，得到最小值的原始表示
            return struct.unpack("!d", struct.pack("!Q", min_raw))[0]  # 转换为双精度浮点数并返回
        else:  # 如果是整数类型
            assert not self.is_signed() or self.size_bits <= 64, (  # 断言有符号整数位数不超过64
                "Cannot represent min as a int64_t"
            )

            if self.is_signed():  # 如果是有符号整数
                return -(1 << (self.size_bits - 1))  # 返回最小值：-2^(size_bits-1)
            else:  # 如果是无符号整数
                return 0  # 无符号整数最小值为0

    @functools.cached_property  # 缓存属性装饰器，计算后缓存结果
    def id(self) -> int:
        """
        Convert the ScalarType to an int which can be passed to pytorch custom
        ops. This layout of the int must be kept in sync with the C++
        ScalarType's from_id method.

        将ScalarType转换为一个整数ID，可以传递给PyTorch自定义算子。
        该整数的位布局必须与C++ ScalarType的from_id方法保持同步。
        """
        val = 0  # 初始化ID值为0
        offset = 0  # 初始化位偏移量为0

        def or_and_advance(member, bit_width):
            """
            将成员值编码到指定位宽中，并按位或到val中，然后推进偏移量。

            参数:
                member: 要编码的成员值
                bit_width: 该成员占用的位宽
            """
            nonlocal val  # 引用外层变量val
            nonlocal offset  # 引用外层变量offset
            bit_mask = (1 << bit_width) - 1  # 创建位掩码，用于截取指定位宽
            val = val | (int(member) & bit_mask) << offset  # 将成员值按位或到val的对应位置
            offset = offset + bit_width  # 推进偏移量

        or_and_advance(self.exponent, 8)  # 编码指数位数（占8位）
        or_and_advance(self.mantissa, 8)  # 编码尾数位数（占8位）
        or_and_advance(self.signed, 1)  # 编码符号标志（占1位）
        or_and_advance(self.bias, 32)  # 编码偏置值（占32位）
        or_and_advance(self._finite_values_only, 1)  # 编码是否仅有限值（占1位）
        or_and_advance(self.nan_repr.value, 8)  # 编码NaN表示方式（占8位）

        assert offset <= 64, f"ScalarType fields too big {offset} to fit into an int64"  # 断言总位数不超过64位

        _SCALAR_TYPES_ID_MAP[val] = self  # 将ID到ScalarType实例的映射存入全局字典

        return val  # 返回计算出的ID

    @property  # 属性装饰器
    def size_bits(self) -> int:
        """
        计算该标量类型的总位数。

        返回:
            int: 指数位 + 尾数位 + 符号位（如果有）
        """
        return self.exponent + self.mantissa + int(self.signed)  # 总位数 = 指数位 + 尾数位 + 符号位

    def min(self) -> int | float:
        """
        Min representable value for this scalar type.
        (accounting for bias if there is one)

        该标量类型可表示的最小值（考虑偏置）。
        """
        return self._raw_min() - self.bias  # 最小值 = 原始最小值 - 偏置

    def max(self) -> int | float:
        """
        Max representable value for this scalar type.
        (accounting for bias if there is one)

        该标量类型可表示的最大值（考虑偏置）。
        """
        return self._raw_max() - self.bias  # 最大值 = 原始最大值 - 偏置

    def is_signed(self) -> bool:
        """
        If the type is signed (i.e. has a sign bit), same as `signed`
        added for consistency with:
        https://pytorch.org/docs/stable/generated/torch.Tensor.is_signed.html

        判断该类型是否有符号（即是否有符号位），与signed属性相同，
        为与PyTorch的is_signed接口保持一致而添加。
        """
        return self.signed  # 返回是否有符号

    def is_floating_point(self) -> bool:
        """
        判断该类型是否为浮点类型。

        返回:
            bool: 指数位不为0则为浮点类型
        """
        "If the type is a floating point type"
        return self.exponent != 0  # 指数位不为0即为浮点类型

    def is_integer(self) -> bool:
        """
        判断该类型是否为整数类型。

        返回:
            bool: 指数位为0则为整数类型
        """
        "If the type is an integer type"
        return self.exponent == 0  # 指数位为0即为整数类型

    def has_bias(self) -> bool:
        """
        判断该类型是否有非零偏置。

        返回:
            bool: 偏置不为0则返回True
        """
        "If the type has a non-zero bias"
        return self.bias != 0  # 偏置不为0即有偏置

    def has_infs(self) -> bool:
        """
        判断该浮点类型是否支持无穷大。

        返回:
            bool: 如果不是仅有限值则支持无穷大
        """
        "If the type is floating point and supports infinity"
        return not self._finite_values_only  # 非仅有限值模式即支持无穷大

    def has_nans(self) -> bool:
        """
        判断该类型是否支持NaN（非数值）。

        返回:
            bool: NaN表示方式不为NONE则支持NaN
        """
        return self.nan_repr != NanRepr.NONE.value  # NaN表示不为NONE即支持NaN

    def is_ieee_754(self) -> bool:
        """
        If the type is a floating point type that follows IEEE 754
        conventions

        判断该类型是否遵循IEEE 754浮点数标准。
        """
        return self.nan_repr == NanRepr.IEEE_754.value and not self._finite_values_only  # NaN表示为IEEE 754且支持无穷大

    def __str__(self) -> str:
        """
        naming generally follows: https://github.com/jax-ml/ml_dtypes
        for floating point types (leading f) the scheme is:
        `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
        flags:
          - no-flags: means it follows IEEE 754 conventions
          - f: means finite values only (no infinities)
          - n: means nans are supported (non-standard encoding)
        for integer types the scheme is:
          `[u]int<size_bits>[b<bias>]`
          - if bias is not present it means its zero

        将标量类型转换为字符串表示。
        浮点类型命名格式: float<总位数>_e<指数位>m<尾数位>[标志]
        整数类型命名格式: [u]int<总位数>[b<偏置>]
        """
        if self.is_floating_point():  # 如果是浮点类型
            ret = (  # 构建浮点类型名称字符串
                "float"
                + str(self.size_bits)  # 添加总位数
                + "_e"
                + str(self.exponent)  # 添加指数位数
                + "m"
                + str(self.mantissa)  # 添加尾数位数
            )

            if not self.is_ieee_754():  # 如果不是IEEE 754标准类型
                if self._finite_values_only:  # 如果仅支持有限值
                    ret = ret + "f"  # 添加'f'标志
                if self.nan_repr != NanRepr.NONE:  # 如果支持NaN
                    ret = ret + "n"  # 添加'n'标志

            return ret  # 返回浮点类型名称
        else:  # 如果是整数类型
            ret = ("int" if self.is_signed() else "uint") + str(self.size_bits)  # 构建整数类型名称（有符号int/无符号uint + 位数）
            if self.has_bias():  # 如果有偏置
                ret = ret + "b" + str(self.bias)  # 添加偏置信息
            return ret  # 返回整数类型名称

    def __repr__(self) -> str:
        """
        返回ScalarType的可打印表示形式。

        返回:
            str: 'ScalarType.' + 类型名称
        """
        return "ScalarType." + self.__str__()  # 返回带前缀的类型名称

    # __len__ needs to be defined (and has to throw TypeError) for pytorch's
    # opcheck to work.
    def __len__(self) -> int:
        """
        定义长度方法（必须抛出TypeError），这是PyTorch的opcheck正常工作所必需的。

        抛出:
            TypeError: 总是抛出，因为ScalarType没有长度概念
        """
        raise TypeError  # 抛出TypeError异常

    #
    # Convenience Constructors
    #

    @classmethod  # 类方法装饰器
    def int_(cls, size_bits: int, bias: int | None) -> "ScalarType":
        """
        Create a signed integer scalar type (size_bits includes sign-bit).

        创建有符号整数标量类型（size_bits包含符号位）。

        参数:
            size_bits: 总位数（包含符号位）
            bias: 偏置值，None时默认为0

        返回:
            ScalarType: 有符号整数标量类型实例
        """
        ret = cls(0, size_bits - 1, True, bias if bias else 0)  # 创建有符号整数类型：指数为0，尾数位=总位数-1（减去符号位）
        ret.id  # noqa B018: make sure the id is cached  # 访问id属性以触发缓存
        return ret  # 返回创建的标量类型

    @classmethod  # 类方法装饰器
    def uint(cls, size_bits: int, bias: int | None) -> "ScalarType":
        """
        Create an unsigned integer scalar type.

        创建无符号整数标量类型。

        参数:
            size_bits: 总位数
            bias: 偏置值，None时默认为0

        返回:
            ScalarType: 无符号整数标量类型实例
        """
        ret = cls(0, size_bits, False, bias if bias else 0)  # 创建无符号整数类型：指数为0，尾数位=总位数，无符号
        ret.id  # noqa B018: make sure the id is cached  # 访问id属性以触发缓存
        return ret  # 返回创建的标量类型

    @classmethod  # 类方法装饰器
    def float_IEEE754(cls, exponent: int, mantissa: int) -> "ScalarType":
        """
        Create a standard floating point type
        (i.e. follows IEEE 754 conventions).

        创建标准IEEE 754浮点类型。

        参数:
            exponent: 指数位数
            mantissa: 尾数位数

        返回:
            ScalarType: 符合IEEE 754标准的浮点标量类型实例
        """
        assert mantissa > 0 and exponent > 0  # 断言尾数和指数位数都大于0
        ret = cls(exponent, mantissa, True, 0)  # 创建IEEE 754浮点类型：有符号，无偏置
        ret.id  # noqa B018: make sure the id is cached  # 访问id属性以触发缓存
        return ret  # 返回创建的标量类型

    @classmethod  # 类方法装饰器
    def float_(
        cls, exponent: int, mantissa: int, finite_values_only: bool, nan_repr: NanRepr
    ) -> "ScalarType":
        """
        Create a non-standard floating point type
        (i.e. does not follow IEEE 754 conventions).

        创建非标准浮点类型（不遵循IEEE 754标准）。

        参数:
            exponent: 指数位数
            mantissa: 尾数位数
            finite_values_only: 是否仅支持有限值
            nan_repr: NaN的表示方式

        返回:
            ScalarType: 非标准浮点标量类型实例
        """
        assert mantissa > 0 and exponent > 0  # 断言尾数和指数位数都大于0
        assert nan_repr != NanRepr.IEEE_754, (  # 断言不是IEEE 754的NaN表示（应使用float_IEEE754构造器）
            "use `float_IEEE754` constructor for floating point types that "
            "follow IEEE 754 conventions"
        )
        ret = cls(exponent, mantissa, True, 0, finite_values_only, nan_repr)  # 创建非标准浮点类型
        ret.id  # noqa B018: make sure the id is cached  # 访问id属性以触发缓存
        return ret  # 返回创建的标量类型

    @classmethod  # 类方法装饰器
    def from_id(cls, scalar_type_id: int):
        """
        根据ID查找并返回对应的ScalarType实例。

        参数:
            scalar_type_id: 标量类型的整数ID

        返回:
            ScalarType: 对应的标量类型实例

        抛出:
            ValueError: 如果ID不存在于映射中
        """
        if scalar_type_id not in _SCALAR_TYPES_ID_MAP:  # 检查ID是否存在于映射字典中
            raise ValueError(f"scalar_type_id {scalar_type_id} doesn't exists.")  # 不存在则抛出ValueError
        return _SCALAR_TYPES_ID_MAP[scalar_type_id]  # 返回对应的ScalarType实例


# naming generally follows: https://github.com/jax-ml/ml_dtypes
# for floating point types (leading f) the scheme is:
#  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
#  flags:
#  - no-flags: means it follows IEEE 754 conventions
#  - f: means finite values only (no infinities)
#  - n: means nans are supported (non-standard encoding)
# for integer types the scheme is:
#  `[u]int<size_bits>[b<bias>]`
#  - if bias is not present it means its zero


class scalar_types:
    """预定义的常用标量类型集合类，包含各种整数和浮点量化类型的实例。"""
    int4 = ScalarType.int_(4, None)  # 4位有符号整数
    uint4 = ScalarType.uint(4, None)  # 4位无符号整数
    int8 = ScalarType.int_(8, None)  # 8位有符号整数
    uint8 = ScalarType.uint(8, None)  # 8位无符号整数
    float8_e4m3fn = ScalarType.float_(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN)  # 8位浮点：4位指数3位尾数，仅有限值，扩展范围NaN
    float8_e5m2 = ScalarType.float_IEEE754(5, 2)  # 8位IEEE 754浮点：5位指数2位尾数
    float8_e8m0fnu = ScalarType(8, 0, False, 0, True, NanRepr.EXTD_RANGE_MAX_MIN)  # 8位特殊浮点：8位指数0位尾数，无符号，仅有限值
    float16_e8m7 = ScalarType.float_IEEE754(8, 7)  # 16位IEEE 754浮点：8位指数7位尾数（即bfloat16）
    float16_e5m10 = ScalarType.float_IEEE754(5, 10)  # 16位IEEE 754浮点：5位指数10位尾数（即float16）

    # fp6, https://github.com/usyd-fsalab/fp6_llm/tree/main
    # and https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    float6_e3m2f = ScalarType.float_(3, 2, True, NanRepr.NONE)  # 6位浮点：3位指数2位尾数，仅有限值，无NaN

    float6_e2m3f = ScalarType.float_(2, 3, True, NanRepr.NONE)  # 6位浮点：2位指数3位尾数，仅有限值，无NaN

    # fp4, https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)  # 4位浮点：2位指数1位尾数，仅有限值，无NaN

    # "gptq" types
    uint2b2 = ScalarType.uint(2, 2)  # 2位无符号整数，偏置为2（GPTQ量化类型）
    uint3b4 = ScalarType.uint(3, 4)  # 3位无符号整数，偏置为4（GPTQ量化类型）
    uint4b8 = ScalarType.uint(4, 8)  # 4位无符号整数，偏置为8（GPTQ量化类型）
    uint8b128 = ScalarType.uint(8, 128)  # 8位无符号整数，偏置为128（GPTQ量化类型）

    # colloquial names
    bfloat16 = float16_e8m7  # bfloat16的别名，即16位浮点（8位指数7位尾数）
    float16 = float16_e5m10  # float16的别名，即16位浮点（5位指数10位尾数）
