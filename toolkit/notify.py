from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib


def send_mail(
    receiver: str, 
    title: str, 
    content: str, 
    sender: str, 
    smtp_user: str, 
    smtp_pass: str, 
    smtp_host: str, 
    smtp_port: int
):
    message = MIMEMultipart("alternative")
    message["Subject"] = title
    message["From"] = sender
    message["To"] = receiver

    part1 = MIMEText(content, "plain")

    message.attach(part1)

    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)

        server.sendmail(sender, receiver, message.as_string())
        server.quit()
    except Exception as e:
        print("Something went wrong...", e)