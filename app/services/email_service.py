"""
Email service for sending emails via Gmail SMTP.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from ..config.settings import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Email service for sending emails via Gmail SMTP."""
    
    @staticmethod
    def create_otp_email_html(otp_code: str, purpose: str = "verification") -> str:
        """Create professional dark-themed HTML email for OTP."""
        purpose_text = {
            "registration": "Welcome to Daddy's AI",
            "login": "Sign in to Your Account",
            "password_reset": "Reset Your Password"
        }.get(purpose, "Verify Your Email")
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Verification Code</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: #000000; padding: 40px 20px;">
    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="max-width: 600px; margin: 0 auto; background: #1a1a1a; border-radius: 16px; overflow: hidden; border: 1px solid #2a2a2a;">
        <!-- Header -->
        <tr>
            <td style="background: #1a1a1a; padding: 40px 30px; text-align: center; border-bottom: 1px solid #2a2a2a;">
                <h1 style="margin: 0 0 8px 0; color: #ffffff; font-size: 28px; font-weight: 600; letter-spacing: -0.5px;">
                    Daddy's AI
                </h1>
                <p style="margin: 0; color: #888888; font-size: 15px;">
                    {purpose_text}
                </p>
            </td>
        </tr>
        
        <!-- Body -->
        <tr>
            <td style="padding: 40px 30px;">
                <p style="margin: 0 0 24px 0; color: #cccccc; font-size: 15px; line-height: 1.6;">
                    Your verification code is:
                </p>
                
                <!-- OTP Code -->
                <div style="background: #252525; border: 1px solid #333333; border-radius: 12px; padding: 24px; text-align: center; margin: 0 0 24px 0;">
                    <div style="font-size: 42px; font-weight: 700; color: #ffffff; letter-spacing: 10px; font-family: 'Courier New', monospace;">
                        {otp_code}
                    </div>
                </div>
                
                <p style="margin: 0 0 16px 0; color: #888888; font-size: 14px; line-height: 1.6;">
                    This code will expire in <strong style="color: #cccccc;">10 minutes</strong>. 
                    Please do not share this code with anyone.
                </p>
                
                <p style="margin: 0; color: #666666; font-size: 13px; line-height: 1.6;">
                    If you didn't request this code, you can safely ignore this email.
                </p>
            </td>
        </tr>
        
        <!-- Footer -->
        <tr>
            <td style="background: #161616; padding: 24px 30px; text-align: center; border-top: 1px solid #2a2a2a;">
                <p style="margin: 0 0 8px 0; color: #888888; font-size: 13px;">
                    <strong style="color: #ffffff;">Daddy's AI</strong>
                </p>
                <p style="margin: 0; color: #666666; font-size: 12px;">
                    Your AI-powered assistant for Indian Markets
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
        """
    
    @staticmethod
    def create_welcome_email_html(user_name: str) -> str:
        """Create professional dark-themed welcome email."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Daddy's AI</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: #000000; padding: 40px 20px;">
    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="max-width: 600px; margin: 0 auto; background: #1a1a1a; border-radius: 16px; overflow: hidden; border: 1px solid #2a2a2a;">
        <!-- Header -->
        <tr>
            <td style="background: #1a1a1a; padding: 50px 30px; text-align: center; border-bottom: 1px solid #2a2a2a;">
                <div style="font-size: 48px; margin-bottom: 16px;">🎉</div>
                <h1 style="margin: 0; color: #ffffff; font-size: 32px; font-weight: 600;">
                    Welcome to Daddy's AI!
                </h1>
            </td>
        </tr>
        
        <!-- Body -->
        <tr>
            <td style="padding: 40px 30px;">
                <h2 style="margin: 0 0 16px 0; color: #ffffff; font-size: 22px; font-weight: 600;">
                    Hi {user_name}! 👋
                </h2>
                
                <p style="margin: 0 0 20px 0; color: #cccccc; font-size: 15px; line-height: 1.6;">
                    Thank you for joining <strong style="color: #ffffff;">Daddy's AI</strong>! 
                    We're excited to help you navigate the Indian markets with AI-powered insights.
                </p>
                
                <div style="background: #252525; border: 1px solid #333333; border-left: 3px solid #ffffff; padding: 20px; margin: 24px 0; border-radius: 8px;">
                    <h3 style="margin: 0 0 12px 0; color: #ffffff; font-size: 16px; font-weight: 600;">What you can do:</h3>
                    <ul style="margin: 0; padding-left: 20px; color: #cccccc; line-height: 1.8; font-size: 14px;">
                        <li>Get real-time NSE/BSE stock insights</li>
                        <li>Analyze crypto market trends</li>
                        <li>Receive technical analysis</li>
                        <li>Access portfolio intelligence</li>
                    </ul>
                </div>
                
                <p style="margin: 24px 0 0 0; color: #cccccc; font-size: 15px; line-height: 1.6;">
                    Ready to get started? Head to your dashboard and ask anything!
                </p>
            </td>
        </tr>
        
        <!-- Footer -->
        <tr>
            <td style="background: #161616; padding: 24px 30px; text-align: center; border-top: 1px solid #2a2a2a;">
                <p style="margin: 0 0 8px 0; color: #888888; font-size: 13px;">
                    <strong style="color: #ffffff;">Daddy's AI</strong>
                </p>
                <p style="margin: 0; color: #666666; font-size: 12px;">
                    Your AI-powered assistant for Indian Markets
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
        """
    
    @staticmethod
    async def send_email(
        to_email: str,
        subject: str,
        html_content: str,
        plain_text: Optional[str] = None
    ) -> bool:
        """Send email via Gmail SMTP."""
        try:
            # Use EMAIL_SERVER_* variables (preferred) or fallback to SMTP_*
            smtp_host = settings.email_server_host or settings.smtp_host
            smtp_port = settings.email_server_port or settings.smtp_port
            smtp_user = settings.email_server_user or settings.smtp_email
            smtp_password = settings.email_server_password or settings.smtp_password
            from_email = settings.email_from or settings.smtp_email
            from_name = settings.smtp_from_name
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{from_name} <{from_email}>"
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add plain text version (fallback)
            if plain_text:
                part1 = MIMEText(plain_text, 'plain')
                msg.attach(part1)
            
            # Add HTML version
            part2 = MIMEText(html_content, 'html')
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    @staticmethod
    async def send_otp_email(email: str, otp_code: str, purpose: str = "registration") -> bool:
        """Send OTP email."""
        html_content = EmailService.create_otp_email_html(otp_code, purpose)
        plain_text = f"Your verification code is: {otp_code}\n\nThis code will expire in 10 minutes."
        
        subject = {
            "registration": "Complete Your Registration - Daddy's AI",
            "login": "Your Login Code - Daddy's AI",
            "password_reset": "Reset Your Password - Daddy's AI"
        }.get(purpose, "Your Verification Code - Daddy's AI")
        
        return await EmailService.send_email(email, subject, html_content, plain_text)
    
    @staticmethod
    async def send_welcome_email(email: str, name: str) -> bool:
        """Send welcome email."""
        html_content = EmailService.create_welcome_email_html(name)
        plain_text = f"Welcome to Daddy's AI, {name}!\n\nThank you for joining us. We're excited to help you navigate the Indian markets."
        
        return await EmailService.send_email(
            email,
            "Welcome to Daddy's AI! 🎉",
            html_content,
            plain_text
        )
