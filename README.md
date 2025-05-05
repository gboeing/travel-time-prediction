# Travel time prediction from sparse open data

## Guidelines

Keep all code in the `code` folder. Code should be type-annotated and use numpy-style docstrings.

Keep all data outside this repo (such as a shared Google Drive) and provide direct links (accessible *only* to people explicitly granted access) in the readme file at `data/README.md`

Ensure every (non standard library) package imported by the code is added to the `environment.yml` dependencies.

## Setup

Set up the conda environment by running `conda create -f environment.yml` from the repo's root.

The pre-commit hooks are in `.pre-commit-config.yaml` and are automatically run by Github as a continuous integration (CI) workflow by `.github/workflows/tests.yml` for every push to main or PR opened on main. We can merge branches' pull requests into main after they pass these CI checks. Install the pre-commit hooks by changing directories to the repo's root and running the command `pre-commit install`. You can then run the hooks manually with "pre-commit run -a" but they will also run on all "git commit" commands after you've installed them.

## Workflow

Clone this repo to your local computer.

Work in branches and open pull requests to merge your code changes into main.

Install the pre-commit hooks locally (see instructions above) and always fix any pre-commit errors (use the line numbers and error messages as hints).

Whenever you do work, first `git pull` the latest code to your local machine.

Commit code changes frequently, and push at the end of every day you do work.

Open a pull request on Github and confirm the CI tests have passed for your PR, then @ ping your code reviewer there when it's all passing and ready for review before merging into the main branch.
