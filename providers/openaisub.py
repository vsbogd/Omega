import logging
import math
import secrets
import threading
import uuid
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc6749 import OAuth2Token

from config import config_get_by_key
import providers
import lib_llm_ext as llm

logger = logging.getLogger(__name__)
AUTHORIZATION_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = "Omega"


class Credentials:
    """In-memory OAuth credentials; expires_in is a lifetime in seconds."""

    def __init__(self, client_id, host_id, access_token, refresh_token, expires_in,
                 id_token=None, scope=None):
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("OAuth response is missing an access token")
        lifetime = float(expires_in)
        if not math.isfinite(lifetime) or lifetime < 0:
            raise ValueError("Invalid OAuth token lifetime")
        self.client_id = client_id
        self.host_id = host_id
        self.id_token = id_token
        # OAuth returns granted scopes as a space-separated string.
        self.scope = scope.split() if isinstance(scope, str) else list(scope or [])
        self.token = OAuth2Token({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": time.time() + lifetime,
            "token_type": "Bearer",
        })

    @property
    def refresh_token(self):
        return self.token.get("refresh_token")

    @property
    def expires_at(self):
        return self.token["expires_at"]

    def is_expired(self):
        """Returns true if token is expired and needs to be refreshed."""
        return self.token.is_expired(leeway=0)

    def api_token(self):
        """Returns the OpenAI subscription token."""
        return self.token["access_token"]


class CallbackHandler(BaseHTTPRequestHandler):
    """Handle a loopback OAuth callback for one pending sign-in attempt."""

    def __init__(self, *args, owner, oauth, host_id, state, verifier, completed, **kwargs):
        self._owner = owner
        self._oauth = oauth
        self._host_id = host_id
        self._state = state
        self._verifier = verifier
        self._completed = completed
        # BaseHTTPRequestHandler processes the request during construction.
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format, *args):
        # The request URL contains the authorization code.
        pass

    def do_GET(self):
        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query, keep_blank_values=True)
        status, message = 400, "Invalid OAuth callback"
        if parsed.path != "/auth/callback":
            status, message = 404, "Not found"
        elif (len(params.get("state", [])) == 1
              and secrets.compare_digest(params["state"][0], self._state)):
            try:
                if "error" in params:
                    raise RuntimeError("OpenAI authorization was denied")
                if len(params.get("code", [])) != 1:
                    raise ValueError("OAuth callback is missing a code")
                client_ids = params.get("client_id", [])
                if self._oauth.client_id == DYNAMIC_CLIENT_ID:
                    if (len(client_ids) != 1 or not client_ids[0].strip()
                            or client_ids[0] == DYNAMIC_CLIENT_ID):
                        raise ValueError("OAuth registration is missing an issued client ID")
                    # The registration placeholder must never be used for exchange.
                    self._oauth.client_id = client_ids[0]
                elif "client_id" in params and client_ids != [self._oauth.client_id]:
                    raise ValueError("OAuth client ID mismatch")
                data = self._oauth.fetch_token(
                    TOKEN_URL,
                    authorization_response=self._oauth.redirect_uri + "?" + parsed.query,
                    state=self._state,
                    code_verifier=self._verifier,
                    resource=OPENAI_API_ENDPOINT,
                    timeout=30,
                )
                credentials = Credentials(self._oauth.client_id,
                                          self._host_id,
                                          data["access_token"],
                                          data.get("refresh_token"),
                                          data["expires_in"],
                                          id_token=data.get("id_token"),
                                          scope=data.get("scope"))
                if not self._owner._auth_cancel.is_set():
                    self._owner._credentials = credentials
                status, message = 200, "Sign-in complete. You may close this window."
            except Exception:
                self._owner._auth_error = "OpenAI sign-in failed; start sign-in again."
                logger.error(self._owner._auth_error)
                status, message = 400, self._owner._auth_error
            finally:
                self._completed.set()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))


class OpenAISub(providers.LLMProvider):
    """LLM integration based on OpenAI subscription."""

    def __init__(self):
        self._credentials = None
        self._provider = None
        self._auth_thread = None
        self._auth_cancel = threading.Event()
        self._auth_error = None
        self._refresh_lock = threading.Lock()

    def start(self) -> None:
        """Starts LLM provider and initiates the OpenAI sign-in process."""
        if self._credentials is not None:
            if self._credentials.is_expired():
                self._refresh_token()
        else:
            self._get_token()

    def stop(self) -> None:
        """Stop and LLM provider and free resources"""
        self._auth_cancel.set()
        if self._auth_thread is not None:
            self._auth_thread.join(timeout=32)
        if self._provider is not None:
            self._provider.stop()
            self._provider = None

    def _get_token(self):
        """Starts OpenAI OAuth 2.0 authentication process. Prints the URL for
        the authentication and starts the background task which handles the
        OAuth HTTP callback in order to get the authorization code. Gets the
        credentials using the code and saves the credentials in the
        self._credentials. This method returns immediately; a daemon thread
        performs the exchange. First sign-in uses dynamic_agent_client;
        subsequent sign-ins reuse the client ID saved in the credentials.
        The generated host identifier is kept in memory with the credentials.
        Failures are retained in self._auth_error and a later start() retries."""
        if self._auth_thread is not None and self._auth_thread.is_alive():
            return
        credentials = self._credentials
        client_id = (credentials.client_id if credentials else None) or DYNAMIC_CLIENT_ID
        host_id = credentials.host_id if credentials else uuid.uuid4().urn
        authorization_params = {}
        if client_id == DYNAMIC_CLIENT_ID:
            authorization_params["agent_name_hint"] = AGENT_NAME_HINT
        if credentials and credentials.id_token:
            authorization_params["id_token_hint"] = credentials.id_token
        verifier = secrets.token_urlsafe(64)
        completed = threading.Event()
        self._auth_cancel.clear()
        self._auth_error = None

        port = int(config_get_by_key("openaisub_callback_port", 1455))
        server = HTTPServer(("127.0.0.1", port), CallbackHandler)
        server.timeout = 0.5
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        oauth = OAuth2Session(
            client_id,
            token_endpoint_auth_method="none",
            redirect_uri=redirect_uri,
            scope="openid offline_access resource.invoke chatgpt.tokens.use.direct",
            code_challenge_method="S256",
        )
        try:
            authorization_url, state = oauth.create_authorization_url(
                AUTHORIZATION_URL, code_verifier=verifier,
                resource=OPENAI_API_ENDPOINT, ext_agent_host_id=host_id,
                **authorization_params,
            )
        except Exception:
            server.server_close()
            oauth.close()
            raise

        server.RequestHandlerClass = partial(
            CallbackHandler, owner=self, oauth=oauth, host_id=host_id, state=state,
            verifier=verifier, completed=completed,
        )

        def receive_callback():
            deadline = time.monotonic() + 300
            try:
                while not completed.is_set() and not self._auth_cancel.is_set():
                    if time.monotonic() >= deadline:
                        self._auth_error = "OpenAI sign-in timed out; start sign-in again."
                        logger.error(self._auth_error)
                        break
                    server.handle_request()
            finally:
                server.server_close()
                oauth.close()

        self._auth_thread = threading.Thread(target=receive_callback, daemon=True)
        try:
            self._auth_thread.start()
        except Exception:
            server.server_close()
            oauth.close()
            raise
        print(f"Continue with ChatGPT: {authorization_url}", flush=True)

    def _refresh_token(self):
        """Refresh the expired token via OAuth API. This method is synchronous."""
        with self._refresh_lock:
            if self._credentials is None or not self._credentials.refresh_token:
                raise RuntimeError("No refresh token available; sign in again")
            if not self._credentials.is_expired():
                return
            with OAuth2Session(
                self._credentials.client_id,
                token_endpoint_auth_method="none",
                token=dict(self._credentials.token),
            ) as oauth:
                data = oauth.refresh_token(TOKEN_URL, timeout=30)
            # Replace credentials only after the entire response is valid.
            self._credentials = Credentials(
                self._credentials.client_id,
                self._credentials.host_id,
                data["access_token"],
                data.get("refresh_token"),
                data["expires_in"],
                id_token=data.get("id_token", self._credentials.id_token),
                scope=data.get("scope", self._credentials.scope),
            )

    def _llm(self) -> llm.AIProvider | None:
        """Returns an instance of llm.AIProvider class which is initialized by
        the authorization token generated via OpenAI subscription. Returns None
        when sign-in process is not finished yet. If token is expired the
        method refresh the token and creates new llm.AIProvider instance."""
        if self._credentials is None:
            return None

        if self._credentials.is_expired():
            if self._provider is not None:
                self._provider.stop()
            self._provider = None
            self._refresh_token()

        if self._provider is None: 
            model = config_get_by_key("openaisub_model", "gpt-6.1")
            self._provider = llm.AIProvider("OpenAISub",
                                            self._credentials.api_token(),
                                            model, OPENAI_API_ENDPOINT)

        return self._provider

    def chat(self, request: providers.LLMRequest) -> providers.LLMResponse:
        """Chat with OpenAI via subscription."""
        provider = self._llm()
        if provider is None:
            return providers.LLMResponse().with_error(
                self._auth_error or "OpenAI sign-in is not finished yet")
        return provider.chat(request)
