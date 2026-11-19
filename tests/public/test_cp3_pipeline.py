import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from assignment.rate_limiter import RateLimitPlugin
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import is_egress_allowed, build_production_plugins


def test_cp3_boundaries(tmp_path):
    async def check():
        limiter = RateLimitPlugin(2, 60)
        async def ask(user):
            return await limiter.on_user_message_callback(invocation_context=SimpleNamespace(user_id=user), user_message=None)
        with patch('assignment.rate_limiter.time.monotonic', return_value=100):
            assert await ask('a') is None
            assert await ask('a') is None
            assert await ask('a') is not None
            assert await ask('b') is None
        with patch('assignment.rate_limiter.time.monotonic', return_value=160):
            assert await ask('a') is None
        assert limiter.blocked_count == 1
    asyncio.run(check())
    assert [p.name for p in build_production_plugins()] == ['rate_limiter', 'input_guardrail', 'output_guardrail']
    for url in ['http://api.vinbank.example', 'https://api.vinbank.example.evil.com', 'https://evil.com', 'https://user@api.vinbank.example', 'https://api.vinbank.example:444']:
        assert not is_egress_allowed(url, 'transfer 500000')
    for payload in ['admin123', 'db.vinbank.internal', '0901234567', 'a@example.com', 'sk-secret-value']:
        assert not is_egress_allowed('https://api.vinbank.example/v1/transfers', payload)
    assert is_egress_allowed('https://api.vinbank.example/v1/transfers', 'transfer 500000')
    audit = AuditLogPlugin()
    audit.record_input(user_id='a', text='account', request_id='1')
    audit.record_input(user_id='a', text='loan', request_id='2')
    audit.record_output(user_id='a', text='ok', request_id='2')
    audit.record_output(user_id='a', text='blocked', blocked=True, layer='input_guardrail', request_id='1')
    assert [r['input'] for r in audit.logs] == ['loan', 'account']
    assert all(r['latency_ms'] >= 0 for r in audit.logs)
    audit.export_json(str(tmp_path / 'audit.json'))
    monitor = MonitoringAlert(total_requests=10, blocked_requests=8, rate_limit_hits=6, judge_checks=2, judge_fails=1)
    assert len(monitor.check_metrics()) == 3
    assert len(monitor.check_metrics()) == 3
    monitor.export_json(str(tmp_path / 'metrics.json'))
