import re
import requests
from langchain_community.document_loaders import BSHTMLLoader
from github import Github
from github import Auth

class Github_Issue:
    def __init__(
        self,
        repo,
        token,

    ):
        # using an access token
        auth = Auth.Token(token)

        # Public Web Github
        g = Github(auth=auth)

        # Replace with the repository owner and name
        self.repo = g.get_repo(repo)


    def get_issues(self, state):
        # Get all open issues, excluding pull requests
        issues = self.repo.get_issues(state=state)
        return issues

    def get_issue(self, ticket):
        if ticket is not None and isinstance(ticket, int):
            issue = self.repo.get_issue(number=ticket)
            return issue
        else:
            raise AssertionError()
            return None

    def get_comments(self, ticket):

        if ticket is not None and isinstance(ticket, int):

            # Get the issue
            issue = self.repo.get_issue(number=ticket)

            # Get the comments
            comments = issue.get_comments()

            comments_contents = "" 

            # process the comments
            for comment in comments:
                # Do not support url parsing at present
                if comment.body != None and comment.body != "" and comment.body != "[](url)":
                    comments_contents = comments_contents + "{"
                    comments_contents = comments_contents + f"Author: {comment.user.login}, "
                    comments_contents = comments_contents + f" Date: {comment.created_at}, "
                    comments_contents = comments_contents + f" Comment: {comment.body}\n"
                    comments_contents = comments_contents + "},"

            return comments_contents
        else:
            raise AssertionError()
            return None


