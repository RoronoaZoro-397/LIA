"""
Jira Ticket bot

This bot is used to create jira tickets from chat. 
"""
__name__ = 'localConfig'
__package__ = 'ringcentral_bot_framework'

import copy
import requests
from requests.auth import HTTPBasicAuth
from jira import JIRA


def botJoinPrivateChatAction(bot, groupId, user, dbAction):
    """
    This is invoked when the bot is added to a private group.
    """
    bot.sendMessage(
        groupId,
        {
            'text': 
            f'''
            **Hello! I am a Jira Automation Bot.**
            If you want to raise a ticket, please format your message as shown below:

            **Required Fields** 
            - issue report (or issue summary): The issue being reported.

            **Optional Fields**
            - **os**: Specify the operating system where the issue was observed.
            - **description**: Add a detailed description of the issue (steps to reproduce, expected vs. actual behavior, etc.).
            - **issuetype**: Specify the type of issue (e.g., Bug, Task). Defaults to "Bug" if not provided.
            - **priority**: Set the priority level (e.g., High, Normal, Low, Critical). Defaults to "Normal" if not provided.

            **Additional Information**
            - **Attachments**: You can attach relevant files or screenshots to help clarify the issue.

            Reply ![:Person]({bot.id}) if you need the instructions
            '''
        }
    )


def botGotPostAddAction(
    bot,
    groupId,
    creatorId,
    user,
    text,
    dbAction,
    handledByExtension,
    event
):
    """
    This is invoked when the user sends a message to the bot.
    """
    import re

    if handledByExtension:
        return

    # Extract attachments from the event payload
    attachments = event.get("body", {}).get("body", {}).get("attachments", []) or []

    # Checking if the message contains issue and OS details
    issue_match = re.search(r'issue\s*report\s*:\s*(.+)', text, re.IGNORECASE)
    if not issue_match:
        issue_match = re.search(r'issue\s*summary\s*(.+)', text, re.IGNORECASE)
    os_match = re.search(r'os\s*:\s*(.+)', text, re.IGNORECASE)


    if f'![:Person]({bot.id})' in text:
        
        if issue_match:
            issue = issue_match.group(1).strip()
            os = os_match.group(1).strip() if os_match else ""

            if os == "win" or os == "window" or os == "windows" or os == "Win" or os == "Window" or os == "Windows":
                os = "win"

            # Extract optional fields
            description_match = re.search(r'description\s*:\s*(.+)', text, re.IGNORECASE)
            issuetype_match = re.search(r'issuetype\s*:\s*(.+)', text, re.IGNORECASE)
            priority_match = re.search(r'priority\s*:\s*(.+)', text, re.IGNORECASE)

            # Use default values if optional fields are not provided
            description = description_match.group(1).strip() if description_match else ""
            issuetype = issuetype_match.group(1).strip().lower().capitalize() if issuetype_match else "Bug"
            priority = priority_match.group(1).strip().lower().capitalize() if priority_match else "Normal"

            

            access_token = bot.token['access_token']

            # Simulate issuing a JIRA ticket with additional fields
            jira_ticket_id = issueJiraTicket(issue, os, creatorId, attachments, access_token, description, issuetype, priority)

            # Notify the user whether the JIRA ticket was created successfully
            if jira_ticket_id:
                jira_ticket_url = f"https://jira.ringcentral.com/browse/{jira_ticket_id}"  # Construct the ticket URL
                bot.sendMessage(
                    groupId,
                    {
                        'text': f'![:Person]({creatorId}), a JIRA ticket has been created for the issue: "{issue}".\n\n'
                                f'**Details:**\n'
                                f'- **Description:** {description}\n'
                                f'- **Issue Type:** {issuetype}\n'
                                f'Ticket ID: "{jira_ticket_id}". [View Ticket]({jira_ticket_url})'
                                
                    }
                )
            else:
                bot.sendMessage(
                    groupId,
                    {
                        'text': f'![:Person]({creatorId}), failed to create a JIRA ticket for the issue: "{issue}".\n'
                                f'Please try again later.'
                    }
                )
        else:
            bot.sendMessage(
                groupId,
                {
                    'text': f'![:Person]({creatorId}), unable to create a JIRA ticket. Ensure that "issue report or issue summary field is there" are included in the message.'
                }
            )



def issueJiraTicket(issue, os, creatorId, attachments, access_token, description, issuetype, priority):
    """
    Creates a JIRA ticket using the JIRA Python library and uploads attachments.
    """
    import requests
    from jira import JIRA
    import os as os_module  # To avoid confusion with the `os` parameter

    # JIRA credentials and connection
    host = "https://jira.ringcentral.com"  # Replace with your JIRA instance URL
    pat = os_module.environ.get("JIRA_PAT")  # Get PAT from environment variable
    if not pat:
        print("Error: JIRA_PAT environment variable not set")
        return None
    project_key = "RCVNC"  # Replace with your project key

    # Create a connection to JIRA
    headers = JIRA.DEFAULT_OPTIONS["headers"].copy()
    headers["Authorization"] = f"Bearer {pat}"
    try:
        jira = JIRA(server=host, options={"headers": headers})

        # Issue data
        # Issue data
        os_title = f'[{os}]' if os != "" else ""
        issue_data = {
            "project": {"key": project_key},
            "summary": f"[UAT]{os_title}{issue}",
            "description": 
            f'''
                {description}
            ''',  # Include detailed description
            "issuetype": {"name": issuetype},  # Dynamically set the issue type (e.g., Bug, Task)
            "priority": {"name": priority},  # Dynamically set the priority (e.g., High, Normal, Low)
        }

        # Create the issue
        new_issue = jira.create_issue(fields=issue_data)
        print("Issue created successfully!")

        # Handle attachments
        for attachment in attachments:
            file_name = attachment['name']
            file_url = attachment['contentUri']

            headers = {
                "Authorization": f"Bearer {access_token}",  # Using the access token from bot instance
            }

            # Download the file from the provided URL
            response = requests.get(file_url, headers=headers, stream=True)
            if response.status_code == 200:
                # Save the file temporarily
                temp_file_path = os_module.path.join("/tmp", file_name)
                print(temp_file_path)
                with open(temp_file_path, 'wb') as f:
                    f.write(response.content)

                # Attach the file to the issue
                with open(temp_file_path, 'rb') as f:
                    jira.add_attachment(issue=new_issue, attachment=f)
                print(f"Attachment '{file_name}' uploaded successfully.")

                # Clean up the temporary file
                os_module.remove(temp_file_path)
            else:
                print(f"Failed to download attachment from {file_url}")

        return new_issue.key  # Return the created issue key

    except Exception as e:
        print("Failed to create issue:", str(e))
        return None