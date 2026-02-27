# Patent Analytics Platform

A Django-based web application for analyzing patent data from Excel files, providing visualizations and insights on patent trends, applicants, IPCs, and inventor networks.

## Table of Contents
- [Overview](#overview)
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

---

## Recent Changes & Rationale

### 1. Code Optimizations (December 2024 - January 2025)

**Problem:** Original O(n²) loops caused 5-15 minute analysis times:
```python
