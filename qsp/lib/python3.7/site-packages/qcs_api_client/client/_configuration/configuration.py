import os
from pathlib import Path
from typing import Optional

from pydantic.main import BaseModel

from qcs_api_client.client._configuration.error import QCSClientConfigurationError
from qcs_api_client.client._configuration.secrets import (
    QCSClientConfigurationSecrets,
    QCSClientConfigurationSecretsCredentials,
)
from qcs_api_client.client._configuration.settings import (
    QCSClientConfigurationSettings,
    QCSClientConfigurationSettingsProfile,
    QCSAuthServer,
)


QCS_BASE_PATH = Path("~/.qcs").expanduser()

DEFAULT_SECRETS_FILE_PATH = QCS_BASE_PATH / "secrets.toml"
DEFAULT_SETTINGS_FILE_PATH = QCS_BASE_PATH / "settings.toml"


class QCSClientConfiguration(BaseModel):
    """A user's settings and secrets along with a specified profile name.

    This class contains a full representation of user specified
    ``QCSClientConfigurationSecrets`` and ``QCSClientConfigurationSettings``, as
    well as a ``profile_name`` which indicates which
    ``QCSClientConfigurationSettingsProfile`` to access within
    ``QCSClientConfigurationSettings.profiles``.

    Typically, clients will simply call ``QCSClientConfiguration.load``,
    to initialize this class from the specified secrets and settings
    paths.
    """

    profile_name: str
    secrets: QCSClientConfigurationSecrets
    settings: QCSClientConfigurationSettings

    @property
    def auth_server(self) -> QCSAuthServer:
        """Returns the configured authorization server.

        ``self.profile.auth_server_name`` serves as key to
        ``QCSClientConfigurationSettings.auth_servers``.

        Returns:
            The specified ``QCSAuthServer``.

        Raises:
             QCSClientConfigurationError: If
                ``QCSClientConfigurationSettings.auth_servers`` does not have
                a value for the authorization server name.
        """
        server = self.settings.auth_servers.get(self.profile.auth_server_name)
        if server is None:
            raise QCSClientConfigurationError(f"no authorization server configured for {self.profile.auth_server_name}")

        return server

    @property
    def credentials(self) -> QCSClientConfigurationSecretsCredentials:
        """Returns the configured ``QCSClientConfigurationSecretsCredentials``

        ``self.profile.credentials_name`` serves as key to
        ``QCSClientConfigurationSecrets.credentials``.

        Returns:
            The specified ``QCSClientConfigurationSecretsCredentials``.

        Raises:
            QCSClientConfigurationError: If
                ``QCSClientConfigurationSettings.credentials`` does not have
                a value for the specified credentials name.
        """
        credentials = self.secrets.credentials.get(self.profile.credentials_name)
        if credentials is None:
            raise QCSClientConfigurationError(f"no credentials available named '{self.profile.credentials_name}'")
        return credentials

    @property
    def profile(self) -> QCSClientConfigurationSettingsProfile:
        """Returns the configured ``QCSClientConfigurationSettingsProfile``.

        `self.profile_name` serves as key to
        ``QCSClientConfigurationSettingsProfile.profiles``.

        Returns:
            The specified ``QCSClientConfigurationSettingsProfile``.

        Raises:
            QCSClientConfigurationError: If
                ``QCSClientConfigurationSettings.profiles`` does not have
                a value for the specified profile name.
        """
        profile = self.settings.profiles.get(self.profile_name)
        if profile is None:
            raise QCSClientConfigurationError(f"no profile available named '{self.profile_name}'")
        return profile

    @classmethod
    def load(
        cls,
        profile_name: Optional[str] = None,
        settings_file_path: Optional[os.PathLike] = None,
        secrets_file_path: Optional[os.PathLike] = None,
    ) -> "QCSClientConfiguration":
        """Loads a fully specified ``QCSClientConfiguration`` from file.

        It evaluates attribute values according to the following precedence:
        argument value > environment variable > default value.

        Args:
            profile_name: [env: QCS_PROFILE_NAME] The name of the profile
                referenced in the fully parsed ``QCSClientConfigurationSettings.profiles``.
                If the profile name does not exist on ``QCSClientConfigurationSettings``,
                ``QCSClientConfiguration.profile`` will raise an error.
                The default value is "default", which may be overridden by
                ``QCSClientConfigurationSettings.default_profile_name``.
            settings_file_path: [env: QCS_SETTINGS_FILE_PATH] The file path
                from which to parse ``QCSClientConfigurationSettings``. This file
                must exist in TOML format. The default value is ``~/.qcs/settings.toml``.
            secrets_file_path: [env: QCS_SECRETS_FILE_PATH] The file path
                from which to parse ``QCSClientConfigurationSecrets``. This file
                must exist in TOML format. The default value is ``~/.qcs/secrets.toml``.

        Returns:
            A fully specified ``QCSClientConfiguration``, which ``QCSAuth`` may use for
            adding refreshed OAuth2 access tokens to outgoing HTTP requests.
        """
        secrets_file_path = secrets_file_path or os.getenv("QCS_SECRETS_FILE_PATH", DEFAULT_SECRETS_FILE_PATH)

        secrets = QCSClientConfigurationSecrets(file_path=secrets_file_path)

        settings_file_path = settings_file_path or os.getenv("QCS_SETTINGS_FILE_PATH", DEFAULT_SETTINGS_FILE_PATH)

        settings = QCSClientConfigurationSettings(file_path=settings_file_path)

        profile_name = profile_name or os.getenv("QCS_PROFILE_NAME", settings.default_profile_name)

        return cls(profile_name=profile_name, secrets=secrets, settings=settings)
