# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试 Uvicorn 访问日志过滤器：验证 UvicornAccessLogFilter 能正确过滤指定路径的访问日志]
"""
Tests for the UvicornAccessLogFilter class.
"""

import logging

from vllm.logging_utils.access_log_filter import (
    UvicornAccessLogFilter,
    create_uvicorn_log_config,
)


# [测试 UvicornAccessLogFilter 类的各种过滤场景，包括排除路径、查询参数、HTTP 方法和状态码]
class TestUvicornAccessLogFilter:
    """Test cases for UvicornAccessLogFilter."""

    # [测试当排除路径列表为空时，所有日志记录都应被允许通过]
    def test_filter_allows_all_when_no_excluded_paths(self):
        """Filter should allow all logs when no paths are excluded."""
        filter = UvicornAccessLogFilter(excluded_paths=[])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/v1/completions", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is True

    # [测试当排除路径为 None 时，所有日志记录都应被允许通过]
    def test_filter_allows_all_when_excluded_paths_is_none(self):
        """Filter should allow all logs when excluded_paths is None."""
        filter = UvicornAccessLogFilter(excluded_paths=None)

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is True

    # [测试配置排除 /health 路径后，该路径的日志记录被正确过滤]
    def test_filter_excludes_health_endpoint(self):
        """Filter should exclude /health endpoint when configured."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is False

    # [测试配置排除 /metrics 路径后，该路径的日志记录被正确过滤]
    def test_filter_excludes_metrics_endpoint(self):
        """Filter should exclude /metrics endpoint when configured."""
        filter = UvicornAccessLogFilter(excluded_paths=["/metrics"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/metrics", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is False

    # [测试不在排除列表中的端点日志应被正常放行]
    def test_filter_allows_non_excluded_endpoints(self):
        """Filter should allow endpoints not in the excluded list."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health", "/metrics"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "POST", "/v1/completions", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is True

    # [测试同时配置多个排除路径时，所有被排除路径的日志均被正确过滤]
    def test_filter_excludes_multiple_endpoints(self):
        """Filter should exclude multiple configured endpoints."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health", "/metrics", "/ping"])

        # Test /health
        record_health = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_health) is False

        # Test /metrics
        record_metrics = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/metrics", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_metrics) is False

        # Test /ping
        record_ping = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/ping", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_ping) is False

    # [测试即使路径带有查询参数，过滤器也能正确排除该路径]
    def test_filter_with_query_parameters(self):
        """Filter should exclude endpoints even with query parameters."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health?verbose=true", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is False

    # [测试过滤器应忽略 HTTP 方法的差异，对排除路径一律过滤]
    def test_filter_different_http_methods(self):
        """Filter should exclude endpoints regardless of HTTP method."""
        filter = UvicornAccessLogFilter(excluded_paths=["/ping"])

        # Test GET
        record_get = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/ping", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_get) is False

        # Test POST
        record_post = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "POST", "/ping", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_post) is False

    # [测试过滤器应忽略 HTTP 状态码的差异，对排除路径一律过滤]
    def test_filter_with_different_status_codes(self):
        """Filter should exclude endpoints regardless of status code."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        for status_code in [200, 500, 503]:
            record = logging.LogRecord(
                name="uvicorn.access",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg='%s - "%s %s HTTP/%s" %d',
                args=("127.0.0.1:12345", "GET", "/health", "1.1", status_code),
                exc_info=None,
            )
            assert filter.filter(record) is False


# [测试 create_uvicorn_log_config 函数，验证生成的日志配置字典结构正确]
class TestCreateUvicornLogConfig:
    """Test cases for create_uvicorn_log_config function."""

    # [测试生成的配置包含所有必需的日志配置键]
    def test_creates_valid_config_structure(self):
        """Config should have required logging configuration keys."""
        config = create_uvicorn_log_config(excluded_paths=["/health"])

        assert "version" in config
        assert config["version"] == 1
        assert "disable_existing_loggers" in config
        assert "formatters" in config
        assert "handlers" in config
        assert "loggers" in config
        assert "filters" in config

    # [测试生成的配置中包含正确的访问日志过滤器定义]
    def test_config_includes_access_log_filter(self):
        """Config should include the access log filter."""
        config = create_uvicorn_log_config(excluded_paths=["/health", "/metrics"])

        assert "access_log_filter" in config["filters"]
        filter_config = config["filters"]["access_log_filter"]
        assert filter_config["()"] == UvicornAccessLogFilter
        assert filter_config["excluded_paths"] == ["/health", "/metrics"]

    # [测试过滤器被正确应用到 access handler 上]
    def test_config_applies_filter_to_access_handler(self):
        """Config should apply the filter to the access handler."""
        config = create_uvicorn_log_config(excluded_paths=["/health"])

        assert "access" in config["handlers"]
        assert "filters" in config["handlers"]["access"]
        assert "access_log_filter" in config["handlers"]["access"]["filters"]

    # [测试自定义日志级别能被正确设置到配置中]
    def test_config_with_custom_log_level(self):
        """Config should respect custom log level."""
        config = create_uvicorn_log_config(
            excluded_paths=["/health"], log_level="debug"
        )

        assert config["loggers"]["uvicorn"]["level"] == "DEBUG"
        assert config["loggers"]["uvicorn.access"]["level"] == "DEBUG"
        assert config["loggers"]["uvicorn.error"]["level"] == "DEBUG"

    # [测试空排除路径列表下配置仍可正常工作]
    def test_config_with_empty_excluded_paths(self):
        """Config should work with empty excluded paths."""
        config = create_uvicorn_log_config(excluded_paths=[])

        assert config["filters"]["access_log_filter"]["excluded_paths"] == []

    # [测试排除路径为 None 时配置仍可正常工作]
    def test_config_with_none_excluded_paths(self):
        """Config should work with None excluded paths."""
        config = create_uvicorn_log_config(excluded_paths=None)

        assert config["filters"]["access_log_filter"]["excluded_paths"] == []


# [集成测试：验证过滤器在真实 Python logger 场景下的工作表现]
class TestIntegration:
    """Integration tests for the access log filter."""

    # [测试过滤器在模拟 uvicorn.access 的真实 logger 中能正确过滤和放行]
    def test_filter_with_real_logger(self):
        """Test filter works with a real Python logger simulating uvicorn."""
        # Create a logger with our filter (simulating uvicorn.access)
        logger = logging.getLogger("uvicorn.access")
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        logger.handlers = []

        # Create a custom handler that tracks messages
        logged_messages: list[str] = []

        class TrackingHandler(logging.Handler):
            def emit(self, record):
                logged_messages.append(record.getMessage())

        handler = TrackingHandler()
        handler.setLevel(logging.INFO)
        filter = UvicornAccessLogFilter(excluded_paths=["/health", "/metrics"])
        handler.addFilter(filter)
        logger.addHandler(handler)

        # Log using uvicorn's format with args tuple
        # Format: '%s - "%s %s HTTP/%s" %d'
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "GET",
            "/health",
            "1.1",
            200,
        )
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "GET",
            "/v1/completions",
            "1.1",
            200,
        )
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "GET",
            "/metrics",
            "1.1",
            200,
        )
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "POST",
            "/v1/chat/completions",
            "1.1",
            200,
        )

        # Verify only non-excluded endpoints were logged
        assert len(logged_messages) == 2
        assert "/v1/completions" in logged_messages[0]
        assert "/v1/chat/completions" in logged_messages[1]

    # [测试非 uvicorn.access 来源的日志不受过滤器影响]
    def test_filter_allows_non_uvicorn_access_logs(self):
        """Test filter allows logs from non-uvicorn.access loggers."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        # Log record from a different logger name
        record = logging.LogRecord(
            name="uvicorn.error",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some error message about /health",
            args=(),
            exc_info=None,
        )

        # Should allow because it's not from uvicorn.access
        assert filter.filter(record) is True

    # [测试过滤器对格式异常的 args 参数的容错处理]
    def test_filter_handles_malformed_args(self):
        """Test filter handles log records with unexpected args format."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        # Log record with insufficient args
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some message",
            args=("only", "two"),
            exc_info=None,
        )

        # Should allow because args doesn't have expected format
        assert filter.filter(record) is True

    # [测试过滤器对 None args 的容错处理]
    def test_filter_handles_non_tuple_args(self):
        """Test filter handles log records with non-tuple args."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        # Log record with None args
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some message without args",
            args=None,
            exc_info=None,
        )

        # Should allow because args is None
        assert filter.filter(record) is True
