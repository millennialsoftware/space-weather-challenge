"""
email_service.py

This module provides a function to send emails using the Yagmail SMTP client. 
It loads email credentials from environment variables and supports optional 
attachments.
"""

# Imports
import yagmail
from dotenv import load_dotenv
import os
from typing import List, Optional, Union

def send_email(subject: str, content: str, attachments: Optional[Union[str, List[str]]] = None) -> None:
    """Sends an email using Yagmail with optional attachments.

    Args:
        subject (str): The subject of the email.
        content (str): The email body (supports plain text or HTML).
        attachments (Optional[Union[str, List[str]]], optional): File path(s) to attach. Defaults to None.

    Raises:
        Exception: If email credentials are missing or there is an issue with sending the email.

    Example:
        send_email(
            subject="Project Update",
            content="<p>Hello, this is an HTML email!</p>",
            attachments=["report.pdf", "image.png"]
        )
    """
    # Load environment variables
    load_dotenv()

    # Email credentials
    sender_email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    recipients = os.getenv("RECIPIENTS", "").split(",")

    if not sender_email or not password:
        raise ValueError("Missing email credentials. Set EMAIL and PASSWORD in environment variables.")

    # Initialize yagmail
    yag = yagmail.SMTP(sender_email, password)

    # Send email with optional attachments
    yag.send(
        to=recipients,  
        subject=subject,
        contents=[content],  
        attachments=attachments if attachments else None  
    )
