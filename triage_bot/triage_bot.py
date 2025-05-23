import os
import argparse
from github_issue import Github_Issue 
from triage import submit_triage_request
import time
# Collect all the issues


def main():
    parser = argparse.ArgumentParser(description='Download artifacts from all open PRs in a GitHub repository (single process)')
    parser.add_argument('--owner', required=True, help='GitHub repository owner')
    parser.add_argument('--repo', required=True, help='GitHub repository name')
    parser.add_argument('--artifact-name', required=True, help='Name of the artifact to download')
    parser.add_argument('--token', required=True, help='GitHub personal access token')
    parser.add_argument('--output-dir', default='./artifacts', help='Output directory for downloaded artifacts')
    parser.add_argument('--delay', type=float, default=60, 
                       help='Delay between API requests in seconds (to avoid rate limiting)')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    repo = args.repo
    owner = args.owner
    token = args.token

    github_issue = Github_Issue(owner + '/' + repo, token)
    issues = github_issue.get_issues("all")
    
    total_artifacts = 0
    successful_downloads = 0
    for issue in issues:
        state = issue.state
        if issue.pull_request != None and state == "open":
            pr_number = issue.number
            #owner = issue.login
            #for testing
            if pr_number != 1672:
                continue
    
            runs = github_issue.get_workflow_runs_for_pr(pr_number, token)
            time.sleep(args.delay)
    
            if not runs:
                print(f"No workflow runs found for PR #{pr_number}")
                continue
    
            artifact_found = False
            artifact_name = args.artifact_name + '-' + str(pr_number) + "-op_regression-op_regression_dev1-op_transformers-op_extended-op_ut-xpu_distributed"

            if len(runs):
                run = runs[0]
                pr_dir = os.path.join(args.output_dir, f"PR-{pr_number}")
                extract_path = os.path.join(pr_dir, f"{artifact_name}-run-{run['id']}")
                failure_list_path = os.path.join(extract_path, "ut_failure_list.csv")

                if os.path.isfile(failure_list_path):
                    print(failure_list_path + " exists, skip downloading!")
                    continue

                artifact = github_issue.get_artifact(run['id'], artifact_name, token)
                if artifact:
                    total_artifacts += 1
                    if github_issue.download_and_extract_artifact(
                        artifact['id'], artifact_name,
                        pr_number, run['id'], args.token, args.output_dir
                    ):
                        successful_downloads += 1
                        artifact_found = True

                if artifact_found and os.path.isfile(failure_list_path):
                    responses = submit_triage_request(failure_list_path, pr_number)
                    result = ""
                    no = 1
                    for response in responses:
                        result = f"{result}\n{no}. {response[1]} {response[2]} got {response[3]} with error message \n```\n{response[4]}\n```\n triage bot result:\n{response[7]}\n"
                        no = no + 1
                        
                    body = "Triage bot UT analaysis result for reference only, please note unique error message only report once:\n" + result
                    github_issue.add_comment(body, pr_number)

                    time.sleep(args.delay)
                else:
                    print(f"No '{args.artifact_name}' artifacts found for PR #{pr_number}")
    
if __name__ == '__main__':
    main()
