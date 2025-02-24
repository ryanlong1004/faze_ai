

## Pull Requests (PRs)

### General Guidelines

- **Use Meaningful Titles**: PR titles should be clear and concise, summarizing the changes.
- **Provide a Detailed Description**: Include background information, the purpose of changes, and any relevant links.
- **Reference Issues**: If the PR is related to an issue, mention it using `fixes/#<issue_number>` or `closes/#<issue_number>`.
- **Break Down Large Changes**: Prefer multiple smaller PRs over a large one to ease review and merging.
- **Follow Code Style Guidelines**: Ensure consistency by following project-specific linting and formatting rules.  Use the included .editorconfig file or `ruff format`
- **Write Unit Tests**: If applicable, include or update tests to cover the changes.  *Ensure all previous tests pass*
- **Keep Commits Clean**: Use meaningful commit messages and squash commits if necessary to maintain a clean history.
- **Ensure CI/CD Passes**: Make sure tests and automated checks pass before requesting a review.
- **Request Reviews**: Assign appropriate reviewers and include a summary of key areas needing attention.

### Reviewing PRs

- **Be Constructive**: Offer helpful feedback and suggest improvements.
- **Test if Necessary**: If changes impact functionality, test locally or in a staging environment.
- **Approve or Request Changes**: Provide clear reasons when requesting modifications.
- **Merge Strategy**: Use `Squash and Merge` for single commits, `Rebase and Merge` for a clean history, and `Merge Commit` for preserving commit history.

## Tagging

### Tagging Best Practices

- **Follow Semantic Versioning**: Use the format `vMAJOR.MINOR.PATCH` (e.g., `v1.2.3`).
  - **Major** (`v2.0.0`): Breaking changes.
  - **Minor** (`v1.1.0`): New features, but backward-compatible.
  - **Patch** (`v1.0.1`): Bug fixes and small improvements.
- **Use Lightweight vs. Annotated Tags**:
  - **Lightweight Tags**: Quick labels for internal reference (`git tag v1.0.0`).
  - **Annotated Tags**: Preferred for releases, including a message (`git tag -a v1.0.0 -m "Release v1.0.0"`).
- **Push Tags to Remote**: Use `git push origin <tag>` to make tags available in the remote repository.

## Releases

### Creating a Release

1. **Draft a Release**:
   - Navigate to the GitHub repository.
   - Click on `Releases` → `Draft a new release`.
2. **Select a Tag**:
   - Choose an existing tag or create a new one following the semantic versioning format.
3. **Provide Release Notes**:
   - Summarize new features, fixes, and improvements.
   - List breaking changes explicitly.
4. **Attach Artifacts (if applicable)**:
   - Upload binaries, compiled files, or other necessary assets.
5. **Publish the Release**:
   - Click `Publish release` to make it available.

### Automating Releases

- **Use GitHub Actions**: Set up CI/CD workflows to automate tagging and releases.
- **Automate Changelogs**: Tools like `github-changelog-generator` or `Release Drafter` can help maintain consistent release notes.
- **Notify Stakeholders**: Announce releases in relevant communication channels (Slack, email, etc.).

## Conclusion

Following these best practices ensures a smooth development workflow, clear versioning, and well-documented releases. Adhering to these guidelines helps maintain project quality, improves collaboration, and enhances traceability.
