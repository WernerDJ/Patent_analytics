# Patent Analytics Platform

A Django-based web application for analyzing patent data from Excel files, providing visualizations and insights on patent trends, applicants, IPCs, and inventor networks.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#Prerequisites)
- [Architecture Evolution](#architecture-evolution)
- [Recent Changes & Rationale](#recent-changes--rationale)
- [Technology Stack](#technology-stack)
- [Deployment](#deployment)
- [Cost Analysis](#cost-analysis)

---

## Overview

This application allows users to upload patent data (Excel files from patent databases) and generates:
- Geographic analysis of patent origins and destinations
- Word clouds from patent abstracts (nouns, verbs, adjectives)
- IPC (International Patent Classification) trend analysis
- Top applicant identification and analysis
- Patent inventor/applicant network graphs
- AI-powered insights using OpenAI

**Target Users:** Patent analysts, researchers, IP attorneys, and technology scouts conducting competitive intelligence.

**Usage Pattern:** 1-2 users per day, ~20 minutes total daily usage.


## Prerequisites

### Essential Requirements

#### 1. Docker & Docker Compose
**All components of this application are containerized.** You must have Docker installed before proceeding.

- **Install Docker Desktop:**
  - **Linux:** Follow the [official Docker Engine installation guide](https://docs.docker.com/engine/install/)
  - **macOS/Windows:** Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)

- **Verify installation:**
```bash
  docker --version
  docker-compose --version
```

- **Why Docker?** 
  - Ensures consistent environment across development and production
  - Bundles Python dependencies, NLTK data, system fonts (CJK support), and matplotlib configuration
  - Fargate deployment requires container images

#### 2. OpenAI API Key (or Alternative AI Provider)
**Required for AI-powered analytics features:**

- **IPC Analysis:** Summarizes patent abstracts within the most frequent IPC classification groups, providing human-readable technical descriptions instead of just raw classification codes
- **Applicant Intelligence:** Generates competitive intelligence reports on top applicants, retrieving company information and translating non-Latin company names (Chinese, Japanese, Korean, etc.)

**Setup:**
1. Create account at [OpenAI Platform](https://platform.openai.com/)
2. Generate API key from [API Keys page](https://platform.openai.com/api-keys)
3. Add to `.env` file: `OPENAI_API_KEY=sk-...`

**Alternative AI Providers:**
The code can be adapted for other providers (Claude API, Google Gemini, local LLMs) by modifying the `get_top_ipcs_AI_defined()` and `get_top_applicants_AI_description()` methods in `pages/analytic_functions.py`.

**Cost:** ~$0.10-0.50 per analysis session with GPT-4o-mini (used for low cost and speed)

#### 3. AWS Account (for Production Deployment)
**Required services:**
- **S3:** Storage for generated plots
- **ECR:** Container registry
- **ECS Fargate:** Container orchestration
- **Secrets Manager:** Secure credential storage

**AWS CLI installation:**
```bash
# Linux/macOS
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure
```

### Optional Requirements

#### 4. Git (for Version Control)
```bash
# Verify installation
git --version

# If not installed:
# Ubuntu/Debian: sudo apt-get install git
# macOS: xcode-select --install
```

#### 5. Python 3.12+ (for Local Development)
Only needed if running outside Docker:
```bash
python --version
pip install -r requirements.txt
```

---
---

## Architecture Evolution

### Initial Architecture (Heroku + Cloudinary)

**Platform:** Heroku  
**File Storage:** Cloudinary  
**Database:** PostgreSQL (Heroku addon)  
**Task Queue:** Celery + Redis  

**Why this was necessary:**
- Heroku has a **30-second request timeout**
- Patent analysis operations take 5-15 minutes
- Celery allowed background processing to avoid timeouts
- Users could submit jobs and poll for results asynchronously

**Problems:**
- Heroku free tier shut down (November 2022)
- Paid Heroku plans expensive for low-traffic apps (~$50+/month)
- Cloudinary free tier limits (25 credits/month)
- Complex architecture (Django + Celery + Redis + PostgreSQL) for simple use case

---

### Attempted Architecture (AWS Lambda + EC2)

**Initial Plan:**
- Host Django frontend on small EC2 instance (t3.micro)
- Execute heavy analytics via AWS Lambda functions
- Store results in S3

**Why this failed:**
- **Lambda memory limit:** 10GB maximum
- **O(n²) algorithmic complexity:** Patent applicant fuzzy matching, network graph generation, and multi-step OpenAI API calls frequently exceeded Lambda limits
- **Stateful requirements:** Analytics functions depend on maintaining state between operations (e.g., cached top IPCs, applicant groupings)
- **Lambda cold starts:** 5-10 second delays unacceptable for user experience

**Alternatives considered:**
- **Google Cloud Functions:** 60-minute timeout (better than Lambda's 15 min) but still serverless overhead, cold starts, and similar pricing (~$20-30/month)
- **Azure Functions:** Unbounded timeout in Premium plan, but expensive (~$370/month minimum) due to always-on instance requirement

**Conclusion:** Serverless functions (Lambda, Cloud Functions, Azure Functions) are fundamentally incompatible with:
1. Long-running computationally intensive operations
2. Stateful multi-step analytics pipelines
3. Low-frequency, high-duration workloads

---

### Final Architecture (AWS Fargate - Simplified)

**Platform:** AWS ECS Fargate  
**File Storage:** AWS S3  
**Database:** SQLite (embedded)  
**Task Queue:** None (synchronous execution)  

**Why Fargate:**
- **No timeout limits:** Gunicorn configured with 900-second (15-minute) timeout
- **Right-sized resources:** 4 vCPU, 16GB RAM matches development environment
- **Pay-per-use pricing:** Only charged for minutes tasks are running (~$2-3/month for compute)
- **No infrastructure management:** No EC2 instances to patch/maintain
- **Containerized:** Consistent environment between dev and prod

**Why simplified (no Celery):**
- Fargate doesn't have Heroku's 30-second timeout constraint
- For 1-2 users/day, synchronous execution is acceptable
- Removes Redis dependency (~$12/month saved)
- Removes separate Celery worker container (~$30/month saved)
- Simpler deployment and debugging

**User experience trade-off:**
- User must keep browser open during 2-5 minute analysis (acceptable for low traffic)
- 30-40 second cold start when container first launches (Fargate task startup)
- After optimizations, most analyses complete in <2 minutes


