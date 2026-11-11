import providers
import lib_llm_ext as llm

class Credentials:

    def is_expired(self):
        """Returns true if token is expired and needs to be refreshed."""
        pass

    def api_token(self):
        """Returns the OpenAI subscription token."""
        pass

class OpenAISub(providers.LLMProvider):
   """LLM integration based on OpenAI subscription."""

    def __init__(self):
        self._credentials = None

    def start(self) -> None:
        """Starts LLM provider and initiates the OpenAI sign-in process."""
        if not self._credentials is  None:
            if self._credentials.is_expired():
                self._refresh_token()
        else:
            self._get_token()

    def stop(self) -> None:
        """Stop and LLM provider and free resources"""
        pass

    def _get_token(self):
        """Starts OpenAI OAuth 2.0 authentication process. Prints the URL for
        the authentication and starts the background task which handles the
        OAuth HTTP callback in order to get the authorization code. Gets the
        credentials using the code and saves the credentials in the
        self.credentials. This method is asynchronous."""
        pass

    def _refresh_token(self):
        """Refresh the expired token via OAuth API. This method is synchronous."""
        pass

    def _llm(self) -> llm.AIProvider | None:
        """Returns an instance of llm.AIProvider class which is initialized by
        the authorization token generated via OpenAI subscription. Returns None
        when sign-in process is not finished yet. If token is expired the
        method refresh the token and creates new llm.AIProvider instance."""
        if self._credentials is None:
            return None

        if self._credentials.is_expired():
            self._provider.stop()
            self._provider = None
            self._refresh_token()

        if self._provider is None: 
            model = config_get_by_key("openaisub_model", "gpt-6.1")
            self._provider = llm.AIProvider("OpenAISub",
                                            self.credentials.api_token(),
                                            model, "https://api.openai.com/v1")

        return self._provider

    def chat(self, request: providers.LLMRequest) -> LLMResponse:
        """Chat with OpenAI via subscription."""
        return self._llm().chat(args)
