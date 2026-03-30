# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者


from vllm.tokenizers import TokenizerLike  # 导入分词器协议类


def _replace_none_with_empty(tokens: list[str | None]):
    """将token列表中的None值替换为空字符串。"""
    for i, token in enumerate(tokens):  # 遍历token列表
        if token is None:  # 如果token为None
            tokens[i] = ""  # 替换为空字符串


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: TokenizerLike,  # 分词器实例
    output_tokens: list[str],  # 输出的token列表
    skip_special_tokens: bool,  # 是否跳过特殊token
    spaces_between_special_tokens: bool,  # 特殊token之间是否添加空格
) -> str:
    """使用添加的编码器将token列表转换为字符串。"""
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    # Performance improvements: avoid repeated attribute and function lookups;
    # localize frequently used objects;

    sub_texts: list[str] = []  # 存储子文本片段的列表
    current_sub_text: list[str] = []  # 当前正在处理的子文本token列表
    convert_tokens_to_string = tokenizer.convert_tokens_to_string  # 缓存token转字符串方法的引用
    added_vocab_set = set(tokenizer.get_added_vocab())  # 获取添加词汇表的集合以加速查找
    all_special_tokens = (  # 获取需要跳过的特殊token集合
        set(tokenizer.all_special_tokens) if skip_special_tokens else ()  # 如果不跳过则为空元组
    )

    for token in output_tokens:  # 遍历输出的token列表
        # Use precomputed set for skip-special check
        if token in all_special_tokens:  # 如果token是需要跳过的特殊token
            continue  # 跳过该token
        if token in added_vocab_set:  # 如果token在添加的词汇表中
            if current_sub_text:  # 如果当前有未处理的子文本
                sub_texts.append(convert_tokens_to_string(current_sub_text))  # 将当前子文本转为字符串并添加
                current_sub_text.clear()  # 清空当前子文本列表
            sub_texts.append(token)  # 直接添加该token
        else:  # 如果是普通token
            current_sub_text.append(token)  # 添加到当前子文本列表
    if current_sub_text:  # 如果还有剩余的未处理子文本
        sub_texts.append(convert_tokens_to_string(current_sub_text))  # 转换并添加
    if spaces_between_special_tokens:  # 如果需要在特殊token之间添加空格
        return " ".join(sub_texts)  # 用空格连接所有子文本
    return "".join(sub_texts)  # 直接连接所有子文本


# 5 is an arbitrary value that should work for all
# tokenizers (bigger = more conservative).
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5  # 增量解码的初始偏移量，5是一个适用于所有分词器的经验值


def convert_prompt_ids_to_tokens(
    tokenizer: TokenizerLike,  # 分词器实例
    prompt_ids: list[int],  # 提示词的token ID列表
    skip_special_tokens: bool = False,  # 是否跳过特殊token
) -> tuple[list[str], int, int]:
    """将提示词ID转换为token，并返回用于增量解码的token列表和偏移量。

    注意：并非所有token都会被转换为字符串，只有增量解码所需的token才会被转换。
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(  # 仅转换末尾部分的token ID
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :],  # 取末尾偏移量+2个token
        skip_special_tokens=skip_special_tokens,  # 传递是否跳过特殊token的参数
    )
    read_offset = len(new_tokens)  # 设置读取偏移为新token的长度
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)  # 计算前缀偏移，不小于0
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]  # 将None token替换为空字符串
    return new_tokens, prefix_offset, read_offset  # 返回token列表、前缀偏移和读取偏移


def convert_ids_list_to_tokens(
    tokenizer: TokenizerLike,  # 分词器实例
    token_ids: list[int],  # 要转换的token ID列表
) -> list[str]:
    """逐个解码输入的token ID。

    Args:
      tokenizer: 模型使用的分词器
      token_ids: 要转换的token列表（Python列表形式）

    Returns:
      token字符串表示的Python列表

    """
    token_str_lst = []  # 存储结果的列表
    for token_id in token_ids:  # 遍历每个token ID
        # use default skip_special_tokens.
        token_str = tokenizer.decode([token_id])  # 将单个token ID解码为字符串
        if token_str is None:  # 如果解码结果为None
            token_str = ""  # 替换为空字符串
        token_str_lst.append(token_str)  # 添加到结果列表
    return token_str_lst  # 返回token字符串列表


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: TokenizerLike,  # 分词器实例
    all_input_ids: list[int],  # 所有输入的token ID列表
    prev_tokens: list[str] | None,  # 前一次迭代的token列表，首次迭代时为None
    prefix_offset: int,  # 前缀偏移量
    read_offset: int,  # 读取偏移量
    skip_special_tokens: bool = False,  # 是否跳过特殊token
    spaces_between_special_tokens: bool = True,  # 特殊token之间是否添加空格
) -> tuple[list[str], str, int, int]:
    """增量式解码输入的token ID，返回新token和新文本。

    如果`prev_tokens`为None，此函数会将输入ID转换为token并返回token和新文本。
    否则，它将返回新增的token和新文本。

    此函数还会返回新的前缀偏移和读取偏移，供下一次迭代使用。

    这些偏移量是为了应对解码中的清理算法，该算法会根据周围的ID决定是否添加空格。

    Args:
        tokenizer: 使用的分词器。
        all_input_ids: 输入的token ID列表。最后一个ID是新的token ID。
        prev_tokens: 前一次迭代的token列表。如果为None，此函数会将输入ID转换为token。
        prefix_offset: 前缀偏移量。
        read_offset: 读取偏移量。
        skip_special_tokens: 是否跳过特殊token。
        spaces_between_special_tokens: 是否在特殊token之间添加空格。
    """
    new_token_id = all_input_ids[-1]  # 获取最新的token ID
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None  # 判断是否为该序列的首次迭代
    if is_first_iter:  # 如果是首次迭代
        (prev_tokens, prefix_offset, read_offset) = convert_prompt_ids_to_tokens(  # 将提示词ID转为token
            tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens  # 排除最后一个新token
        )
    assert prev_tokens is not None  # 断言prev_tokens不为None

    # If the new token id is out of bounds, return an empty string.
    if 0 <= new_token_id < len(tokenizer):  # 如果新token ID在有效范围内
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(  # 将新token ID转为token
            [new_token_id], skip_special_tokens=skip_special_tokens  # 传入列表以支持跳过特殊token
        )
        if isinstance(new_tokens, str):  # 如果返回的是字符串
            new_tokens = [new_tokens]  # 包装为列表
        else:  # 否则
            # This is required to guard against out-of-vocab prompt token ids
            # (for example when using dummy weights)
            _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]  # 替换None为空字符串
    else:  # 如果token ID超出范围
        new_tokens = [""]  # 返回空字符串token
    output_tokens = prev_tokens + new_tokens  # 合并之前的token和新token

    # If this is the first iteration, return all tokens.
    if is_first_iter:  # 如果是首次迭代
        new_tokens = output_tokens  # 返回所有token

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():  # 如果是快速分词器或没有添加的词汇
        prefix_text = tokenizer.convert_tokens_to_string(  # 将前缀token转为文本
            output_tokens[prefix_offset:read_offset]  # 取前缀偏移到读取偏移之间的token
        )
        new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])  # 将前缀偏移之后的所有token转为文本
    else:  # 如果不是快速分词器且有添加的词汇
        prefix_text = _convert_tokens_to_string_with_added_encoders(  # 使用添加编码器转换前缀文本
            tokenizer,
            output_tokens[prefix_offset:read_offset],  # 前缀范围的token
            skip_special_tokens=skip_special_tokens,  # 是否跳过特殊token
            spaces_between_special_tokens=spaces_between_special_tokens,  # 特殊token间是否加空格
        )
        new_text = _convert_tokens_to_string_with_added_encoders(  # 使用添加编码器转换新文本
            tokenizer,
            output_tokens[prefix_offset:],  # 前缀偏移之后的所有token
            skip_special_tokens=skip_special_tokens,  # 是否跳过特殊token
            spaces_between_special_tokens=spaces_between_special_tokens,  # 特殊token间是否加空格
        )

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):  # 如果新文本不比前缀长，或以替换字符结尾
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        return new_tokens, "", prefix_offset, read_offset  # 返回空文本，保持偏移不变

    new_text = new_text[len(prefix_text) :]  # 去除前缀文本，得到真正的新增文本
    return new_tokens, new_text, read_offset, len(output_tokens)  # 返回新token、新文本和更新后的偏移量
