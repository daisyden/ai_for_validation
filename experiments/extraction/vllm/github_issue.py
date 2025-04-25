import re
import requests
from langchain_community.document_loaders import BSHTMLLoader
from github import Github
from github import Auth
import time
import os
import zipfile


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
        self.token = token
        self.repo_name = repo


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


    def get_workflow_runs_for_pr(self, pr_number, token):
        """Get workflow runs for a specific pull request"""
        url = f"https://api.github.com/repos/{self.repo_name}/actions/runs?event=pull_request&per_page=100"
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            runs = response.json()['workflow_runs']
            return [run for run in runs if 'pull_requests' in run and len(run['pull_requests']) != 0 and run['pull_requests'][0]['number'] == pr_number]
        except Exception as e:
            print(f"Error fetching workflow runs for PR #{pr_number}: {str(e)}")
            return []
    
    def get_artifact(self, run_id, artifact_name, token):
        """Get artifact details for a specific workflow run"""
        url = f"https://api.github.com/repos/{self.repo_name}/actions/runs/{run_id}/artifacts"
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            artifacts = response.json()['artifacts']
            return next((a for a in artifacts if a['name'] == artifact_name), None)
        except Exception as e:
            print(f"Error fetching artifacts for run {run_id}: {str(e)}")
            return None
    
    def download_and_extract_artifact(self, artifact_id, artifact_name, pr_number, run_id, token, output_dir):
        """Download and extract a specific artifact"""
        url = f"https://api.github.com/repos/{self.repo_name}/actions/artifacts/{artifact_id}/zip"
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        pr_dir = os.path.join(output_dir, f"PR-{pr_number}")
        os.makedirs(pr_dir, exist_ok=True)
        zip_path = os.path.join(pr_dir, f"{artifact_name}-run-{run_id}.zip")
        extract_path = os.path.join(pr_dir, f"{artifact_name}-run-{run_id}")
        
        try:
            print(f"Downloading {artifact_name} for PR #{pr_number}, run {run_id}...")
            
            # Download the zip file
            with requests.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract the zip file
            print(f"Extracting {artifact_name} for PR #{pr_number}, run {run_id}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Remove the zip file
            os.remove(zip_path)
            
            print(f"Successfully processed {artifact_name} for PR #{pr_number}, run {run_id}")
            return True
        except Exception as e:
            print(f"Error processing artifact for PR #{pr_number}, run {run_id}: {str(e)}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False
    
    
