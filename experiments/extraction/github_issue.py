import re
import requests
from langchain_community.document_loaders import BSHTMLLoader
from github import Github

class Github_Issue:
    def __init__(
        self,
        repo,
        token,

    ):
        self.repo = repo
        self.token = token

    def get(self, ticket):
        if ticket is not None and isinstance(ticket, int):
            link = "https://github.com/" + self.repo + "/issues/" + str(ticket) 
            response = requests.get(link)
            
            # Write it to a file
            with open("issue.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            
            # Load it with an HTML parser
            loader = BSHTMLLoader("issue.html")
            document = loader.load()[0]

            # Clean up code
            # Replace consecutive new lines with a single new line
            document.page_content = re.sub("\n\n+", "\n", document.page_content)

            return document.page_content
        else:
            raise AssertionError()
            return None

    def get_comments(self, ticket):

        if ticket is not None and isinstance(ticket, int):
            link = "https://github.com/" + self.repo + "/issues/" + str(ticket) 
            
            # Authenticate with GitHub
            github = Github(self.token)
            
            # Get the repository
            repo = github.get_repo(self.repo)
            
            # Get the issue
            issue = repo.get_issue(number=ticket)
            
            # Get the comments
            comments = issue.get_comments()
            
            comments_contents = "Comments of https://github.com/intel/torch-xpu-ops/issues/1437: {\n"
            
            # process the comments
            for comment in comments:
                comments_contents = comments_contents + "{"
                comments_contents = comments_contents + f"Author: {comment.user.login}, "
                comments_contents = comments_contents + f" Date: {comment.created_at}, "
                comments_contents = comments_contents + f" Comment: {comment.body}\n"
                comments_contents = comments_contents + "},"
            comments_contents = comments_contents + " }"
            return comments_contents
        else:
            raise AssertionError()
            return None


       
       
