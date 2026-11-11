"""OAuth tests isolated from the optional LLM runtime dependencies."""
import base64
import hashlib
import importlib.util
import io
import json
import uuid
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from requests import Response
from requests.exceptions import ConnectionError
from authlib.integrations.base_client.errors import OAuthError
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import urlopen


def load_provider():
    spec = importlib.util.spec_from_file_location(
        'openaisub_test_module', Path(__file__).resolve().parents[1] / 'providers/openaisub.py')
    module = importlib.util.module_from_spec(spec)
    stubs = {
        'providers': types.SimpleNamespace(LLMProvider=object, LLMRequest=object,
                                          LLMResponse=object),
        'lib_llm_ext': types.SimpleNamespace(AIProvider=object),
        'config': types.SimpleNamespace(config_get_by_key=lambda key, default=None: default),
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class OAuthTests(unittest.TestCase):
    def setUp(self):
        self.module = load_provider()
        self.provider = self.module.OpenAISub()
        self.module.config_get_by_key = lambda key, default=None: {
            'openaisub_callback_port': 0,
        }.get(key, default)
        transport = patch('requests.sessions.Session.send')
        self.send = transport.start()
        self.addCleanup(transport.stop)
        self.addCleanup(self.provider.stop)
        self.respond({'access_token': 'access', 'refresh_token': 'refresh',
                      'token_type': 'Bearer', 'expires_in': 60})

    def respond(self, token, status=200):
        response = Response()
        response.status_code = status
        response._content = json.dumps(token).encode()
        self.send.return_value = response

    def sent_form(self):
        return {key: values[0] for key, values in
                parse_qs(self.send.call_args.args[0].body).items()}

    def test_authorization_host_id_is_stored_in_credentials(self):
        params = self.begin()
        host_id = params['ext_agent_host_id'][0]
        self.assertEqual(uuid.UUID(host_id).version, 4)
        self.callback(params, code='code', state=params['state'][0])
        self.assertEqual(self.provider._credentials.host_id, host_id)

    def test_callback_stores_id_token_and_granted_scope(self):
        self.respond({'access_token': 'access', 'expires_in': 60,
                      'id_token': 'identity', 'scope': 'openid  resource.invoke'})
        params = self.begin()
        self.assertNotIn('id_token_hint', params)
        self.callback(params, code='code', state=params['state'][0])
        self.assertEqual(self.provider._credentials.id_token, 'identity')
        self.assertEqual(self.provider._credentials.scope, ['openid', 'resource.invoke'])

    def test_reauthorization_uses_saved_identity_and_client(self):
        self.provider._credentials = self.module.Credentials(
            'saved-client', 'host', 'access', 'refresh', 0,
            id_token='saved-identity', scope='openid')
        params = self.begin()
        self.assertEqual(params['id_token_hint'], ['saved-identity'])
        self.assertEqual(params['client_id'], ['saved-client'])
        self.assertEqual(params['ext_agent_host_id'], ['host'])
        self.assertNotIn('agent_name_hint', params)
        self.callback(params, code='code', state=params['state'][0])
        self.assertEqual(self.sent_form()['client_id'], 'saved-client')

    def test_refresh_updates_or_preserves_identity_and_scope(self):
        for extra, expected_id, expected_scope in (
            ({}, 'old-identity', ['openid', 'resource.invoke']),
            ({'id_token': 'new-identity', 'scope': 'openid'}, 'new-identity', ['openid']),
            ({'scope': ''}, 'old-identity', []),
        ):
            with self.subTest(extra=extra):
                self.provider._credentials = self.module.Credentials(
                    'client', 'host', 'old', 'refresh', 0,
                    id_token='old-identity', scope='openid resource.invoke')
                self.respond({'access_token': 'new', 'expires_in': 60, **extra})
                self.provider._refresh_token()
                self.assertEqual(self.provider._credentials.id_token, expected_id)
                self.assertEqual(self.provider._credentials.scope, expected_scope)

    def test_missing_identity_and_scope_have_empty_defaults(self):
        params = self.begin()
        self.assertNotIn('id_token_hint', params)
        self.callback(params, code='code', state=params['state'][0])
        self.assertIsNone(self.provider._credentials.id_token)
        self.assertEqual(self.provider._credentials.scope, [])

    def test_first_sign_in_registers_and_exchanges_issued_client_id(self):
        params = self.begin()
        self.assertEqual(params['client_id'], ['dynamic_agent_client'])
        self.assertEqual(params['scope'][0].split(),
                         ['openid', 'offline_access', 'resource.invoke',
                          'chatgpt.tokens.use.direct'])
        self.assertEqual(params['agent_name_hint'], ['Omega'])
        self.callback(params, code='code', state=params['state'][0], client_id='issued-client')
        self.assertEqual(self.sent_form()['client_id'], 'issued-client')
        self.assertEqual(self.provider._credentials.client_id, 'issued-client')

    def test_registration_rejects_missing_or_invalid_issued_client_id(self):
        for value in (None, '', 'dynamic_agent_client', ['one', 'two']):
            with self.subTest(client_id=value):
                params = self.begin()
                query = {'code': 'code', 'state': params['state'][0]}
                if value is not None:
                    query['client_id'] = value
                url = params['redirect_uri'][0] + '?' + urlencode(query, doseq=True)
                with self.assertRaises(HTTPError) as error:
                    urlopen(url, timeout=3)
                error.exception.close()
                self.provider._auth_thread.join(2)
                self.send.assert_not_called()
                self.assertIsNone(self.provider._credentials)

    def test_reauthorization_rejects_changed_client_id(self):
        old = self.module.Credentials('saved-client', 'host', 'access', 'refresh', 60)
        self.provider._credentials = old
        params = self.begin()
        with self.assertRaises(HTTPError) as error:
            self.callback(params, code='code', state=params['state'][0], client_id='other-client')
        error.exception.close()
        self.send.assert_not_called()
        self.assertIs(self.provider._credentials, old)

    def test_credentials_expiry(self):
        with patch.object(self.module.time, 'time', return_value=100):
            credentials = self.module.Credentials('client', 'host', 'access', 'refresh', 10)
            self.assertEqual(credentials.client_id, 'client')
            self.assertEqual(credentials.host_id, 'host')
            self.assertEqual(credentials.api_token(), 'access')
            self.assertFalse(credentials.is_expired())
        with patch.object(self.module.time, 'time', return_value=110.001):
            self.assertTrue(credentials.is_expired())
        for lifetime in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.module.Credentials('client', 'host', 'access', 'refresh', lifetime)

    def begin(self):
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            self.provider._get_token()
        return parse_qs(urlsplit(output.getvalue().split(' ', 3)[3].strip()).query)

    def callback(self, params, **query):
        if params['client_id'] == ['dynamic_agent_client']:
            query.setdefault('client_id', 'test-client')
        url = params['redirect_uri'][0] + '?' + urlencode(query, doseq=True)
        with urlopen(url, timeout=3) as response:
            return response.status

    def test_callback_state_pkce_and_duplicate_start(self):
        params = self.begin()
        thread = self.provider._auth_thread
        self.provider._get_token()
        self.assertIs(self.provider._auth_thread, thread)
        self.assertIsNone(self.provider._credentials)
        with self.assertRaises(HTTPError) as error:
            self.callback(params, code='code', state='wrong')
        self.assertEqual(error.exception.code, 400)
        self.send.assert_not_called()
        self.assertEqual(self.callback(params, code='code', state=params['state'][0]), 200)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.provider._credentials.client_id, 'test-client')
        self.assertEqual(self.provider._credentials.api_token(), 'access')
        request = self.sent_form()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(
            request['code_verifier'].encode()).digest()).rstrip(b'=').decode()
        self.assertEqual(challenge, params['code_challenge'][0])
        self.assertEqual(request['redirect_uri'], params['redirect_uri'][0])

    def test_denied_callback_and_retry(self):
        params = self.begin()
        with self.assertRaises(HTTPError):
            self.callback(params, error='access_denied', state=params['state'][0])
        self.provider._auth_thread.join(2)
        self.assertIsNone(self.provider._credentials)
        self.assertIsNotNone(self.provider._auth_error)
        self.send.assert_not_called()
        self.begin()
        self.assertIsNone(self.provider._auth_error)

    def test_refresh_rotation_and_omitted_refresh_token(self):
        for extra in ({'refresh_token': 'rotated'}, {}):
            self.provider._credentials = self.module.Credentials('client', 'host', 'old', 'refresh', 0)
            self.respond({'access_token': 'new', 'expires_in': 60, **extra})
            self.provider._refresh_token()
            self.assertEqual(self.provider._credentials.client_id, 'client')
            self.assertEqual(self.provider._credentials.host_id, 'host')
            self.assertEqual(self.provider._credentials.api_token(), 'new')
            self.assertEqual(self.provider._credentials.refresh_token,
                             extra.get('refresh_token', 'refresh'))
            self.assertEqual(self.sent_form(), {
                'grant_type': 'refresh_token', 'client_id': 'client', 'refresh_token': 'refresh'})

    def test_failed_refresh_preserves_credentials(self):
        old = self.module.Credentials('client', 'host', 'old', 'refresh', 0)
        self.provider._credentials = old
        self.send.side_effect = ConnectionError('network failed')
        with self.assertRaises(ConnectionError):
            self.provider._refresh_token()
        self.assertIs(self.provider._credentials, old)
        self.send.side_effect = None
        for response, status, error in (
            ({'error': 'invalid_grant'}, 400, OAuthError),
            ({'access_token': '', 'expires_in': 60}, 200, ValueError),
            ({'access_token': 'new'}, 200, KeyError),
        ):
            self.respond(response, status)
            with self.assertRaises(error):
                self.provider._refresh_token()
            self.assertIs(self.provider._credentials, old)

    def test_exchange_form_and_timeout(self):
        params = self.begin()
        self.callback(params, code='a+b', state=params['state'][0])
        request = self.send.call_args.args[0]
        form = self.sent_form()
        self.assertEqual(form['code'], 'a+b')
        self.assertEqual(form['client_id'], 'test-client')
        self.assertEqual(form['grant_type'], 'authorization_code')
        self.assertEqual(form['resource'], self.module.OPENAI_API_ENDPOINT)
        self.assertEqual(request.url, self.module.TOKEN_URL)
        self.assertEqual(request.method, 'POST')
        self.assertNotIn('client_secret', form)
        self.assertNotIn('Authorization', request.headers)
        self.assertEqual(self.send.call_args.kwargs['timeout'], 30)

    def test_missing_and_duplicate_state_are_ignored(self):
        params = self.begin()
        for query in ({'code': 'code'},
                      {'code': 'code', 'state': ['wrong', params['state'][0]]}):
            url = params['redirect_uri'][0] + '?' + urlencode(query, doseq=True)
            with self.assertRaises(HTTPError) as error:
                urlopen(url, timeout=3)
            error.exception.close()
        self.send.assert_not_called()
        self.assertTrue(self.provider._auth_thread.is_alive())
        self.assertEqual(self.callback(params, code='code', state=params['state'][0]), 200)

    def test_exchange_error_allows_retry(self):
        self.respond({'error': 'invalid_grant'}, 400)
        params = self.begin()
        with self.assertRaises(HTTPError) as error:
            self.callback(params, code='code', state=params['state'][0])
        error.exception.close()
        self.provider._auth_thread.join(2)
        self.assertIsNone(self.provider._credentials)
        self.assertIsNotNone(self.provider._auth_error)
        self.respond({'access_token': 'new', 'expires_in': 60})
        params = self.begin()
        self.callback(params, code='new-code', state=params['state'][0])
        self.assertEqual(self.provider._credentials.api_token(), 'new')

    def test_stop_closes_callback_listener(self):
        self.begin()
        self.provider.stop()
        self.assertFalse(self.provider._auth_thread.is_alive())
        self.send.assert_not_called()

    def test_unexpired_credentials_do_not_refresh(self):
        self.provider._credentials = self.module.Credentials('client', 'host', 'access', 'refresh', 60)
        self.provider._refresh_token()
        self.send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
