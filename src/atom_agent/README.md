# ATOM Agent: Autonomous Task Orchestration & Management

ATOM is a highly structured, stateful AI agent built on LangGraph. It is designed for multi-step software engineering tasks with a focus on auditability, security, and deterministic verification.

## 📖 Documentation Index

| Document                                 | Description                                                                 |
| :--------------------------------------- | :-------------------------------------------------------------------------- |
| [Architecture](./docs/INFRASTRUCTURE.md) | High-level graph structure, state management, and memory tiers.             |
| [Nodes](./docs/NODES.md)                 | Detailed logic and flow for setup, planning, execution, and reflection.     |
| [Tools](./docs/TOOLS.md)                 | Specification for file, code, and sandbox operations with security details. |

## 🚀 Vision

ATOM solves the problem of "Agent Drift" by enforcing a strict **Workspace Contract**. Instead of a chaotic long-running conversation, ATOM breaks work into discrete steps, each with a mandatory "Clinical Audit" (Reflection) and "Authority Promotion" (Commit).

## 🛠 Features

- **Clinical Audit**: Every step is scored against explicit acceptance criteria.
- **Staging-Commit Pattern**: Code is jailed in an attempt folder until verified.
- **Path Hardening**: Zero-trust toolbelt that prevents path escapes or absolute path hijacking.
- **Tiered Memory**: context reconstruction using Cross-Encoder ranking for historical relevance.
- **Bubblewrap Sandbox**: Secure, isolated execution environment for tests and scripts.

---

## 🏗 Key Components at a Glance

### Nodes
- **Planner**: Generates the roadmap.
- **Executor**: Conducts the work.
- **Reflector**: Audits the results.
- **Commit**: Makes results "Authoritative".

### Safeguards
- **Identity Enforcement**: Agents cannot misname folders or step IDs.
- **Rollback Logic**: Fundamental re-planning when tactical refinements fail.
- **Complete Reset**: Rolling back wipes the disk to prevent legacy pollution.
