"""Built-in feedback connectors."""

from .files import FileConnector
from .http import AppleReviewsConnector, GenericJSONConnector, GooglePlayConnector, IntercomConnector, TypeformConnector, ZendeskConnector
from .synthetic import SyntheticConnector

__all__ = ["AppleReviewsConnector", "FileConnector", "GenericJSONConnector", "GooglePlayConnector", "IntercomConnector", "SyntheticConnector", "TypeformConnector", "ZendeskConnector"]
