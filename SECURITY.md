<!-- Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/ -->
# Security Policy

## Reporting a vulnerability
Please report security issues privately via GitHub Security Advisories on this
repository, or by contacting the maintainer
[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/). Do **not** open a public
issue for undisclosed vulnerabilities.

We aim to acknowledge reports within a few business days.

## Scope notes
This project is **shadow-mode / offline-replay only** in v1: it emits
recommendations, it does not actuate production schedulers or facility controls.
The exact symbolic kernel is the safety shield; any change that lets the learned
model bypass kernel feasibility is treated as a security-relevant defect.

## Supported versions
The latest tagged release on `main` receives fixes.
