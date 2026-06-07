import os
import sys
import json
import time
import argparse
from datetime import datetime
from github import Github, GithubException, RateLimitExceededException

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch public repositories for a GitHub user.")
    parser.add_argument("username", type=str, help="GitHub username to fetch repos for")
    parser.add_argument("--output-dir", type=str, default="data/repos", help="Directory to store ingested repo documents")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token (optional, or set GITHUB_TOKEN environment variable)")
    return parser.parse_args()

def handle_rate_limit(g):
    """Checks the rate limit status and sleeps if limit is hit."""
    try:
        rate_limit = g.get_rate_limit().rate
        print(f"[Log] Rate Limit Info: {rate_limit.remaining}/{rate_limit.limit} remaining. Resets at {rate_limit.reset}")
        if rate_limit.remaining == 0:
            reset_time = rate_limit.reset.timestamp()
            sleep_time = max(reset_time - time.time(), 0) + 10  # add 10 seconds buffer
            print(f"[Warning] Rate limit hit. Sleeping for {sleep_time:.2f} seconds until reset...")
            time.sleep(sleep_time)
    except Exception as e:
        print(f"[Warning] Failed to check rate limit: {e}")

def run_with_retry(g, func, *args, **kwargs):
    """Executes a function and retries it if rate limited."""
    while True:
        try:
            return func(*args, **kwargs)
        except RateLimitExceededException:
            print("[Warning] Rate Limit Exceeded Exception caught. Waiting for reset...")
            handle_rate_limit(g)
        except GithubException as e:
            if e.status == 403 and "rate limit" in str(e).lower():
                print("[Warning] 403 Rate Limit Error caught. Waiting for reset...")
                handle_rate_limit(g)
            else:
                raise e

def main():
    args = parse_args()
    
    # Authenticate (Token could come from cli argument or env)
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if token:
        print("[Log] Initializing GitHub client with authentication token.")
        g = Github(token)
    else:
        print("[Log] Initializing GitHub client without authentication (subject to lower rate limits).")
        g = Github()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Log] Saving outputs to directory: {args.output_dir}")

    print(f"[Log] Fetching repositories for user: {args.username}")
    
    try:
        # Check rate limit initially
        handle_rate_limit(g)
        
        user = run_with_retry(g, g.get_user, args.username)
        repos = run_with_retry(g, user.get_repos)
        
        repo_count = 0
        success_count = 0
        
        # Iterate over repositories (handles pagination lazily)
        for repo in repos:
            repo_count += 1
            print(f"\n[Log] Processing repository {repo_count}: {repo.name}")
            
            # Fetch metadata
            try:
                name = repo.name
                description = repo.description or ""
                language = repo.language or ""
                stars = repo.stargazers_count
                
                # Fetch topics (may require extra API call)
                topics = run_with_retry(g, repo.get_topics)
                
                # Fetch README
                readme_content = ""
                try:
                    readme = run_with_retry(g, repo.get_readme)
                    readme_content = readme.decoded_content.decode("utf-8")
                except GithubException as e:
                    if e.status == 404:
                        print(f"[Info] No README found for {repo.name}.")
                    else:
                        print(f"[Warning] Failed to fetch README for {repo.name}: {e}")
                
                # Assemble document
                doc = {
                    "metadata": {
                        "name": name,
                        "description": description,
                        "topics": topics,
                        "language": language,
                        "stars": stars,
                        "html_url": repo.html_url
                    },
                    "readme_content": readme_content
                }
                
                # Save as JSON document
                output_file = os.path.join(args.output_dir, f"{name}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)
                
                print(f"[Log] Successfully stored document for {repo.name} to {output_file}")
                success_count += 1
                
            except Exception as e:
                print(f"[Error] Failed to process repository {repo.name if 'repo' in locals() else 'unknown'}: {e}")
                
        print(f"\n[Log] Completed. Successfully processed {success_count}/{repo_count} repositories.")
        
    except GithubException as e:
        print(f"[Fatal] GitHub API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Fatal] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
