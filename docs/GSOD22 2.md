# Google Season of Docs 2022

## Docs Audit, Update & Expansion

### About OpenMined

OpenMined began in 2017 and is an open-source and not-for-profit community of over 12,000 academics, engineers, mentors, educators, ethicists and privacy enthusiasts committed to making a fairer and more prosperous world by building and teaching privacy enhancing technologies. PySyft (current version 0.6.0, first release 2020) is an Apache 2.0 licensed library for secure and private Deep Learning in Python.

OpenMined continues to build and curate an ecosystem of open-source privacy enhancing tools that span techniques including homomorphic encryption, secure enclaves, federated learning, differential privacy, zero knowledge proofs and much more. We work to solve real-world privacy problems for a variety of important use cases.

The lack of PySyft documentation is making it very difficult to onboard new developers onto our teams, who are currently forced to read the codebase manually without proper direction or guidance.

We have developed and published privacy courses available at [courses.openmined.org](https://courses.openmined.org) in collaboration with the United Nations, Oxford University and PyTorch. Our guest speakers include privacy technology titans including Cynthia Dwork (co-inventor of Differential Privacy), Helen Nissenbaum (Context Integrity), Sameer Wagh (lead author of FALCON) and Glen Weyl (Author of Radical Markets and Quadratic Voting).

We are currently investigating projects that will provide great social impact for the world. There is no time like the present to get involved with OpenMined - join our community at [slack.openmined.org](https://slack.openmined.org).

### About Our Project

OpenMined is dedicated to solving various problems in the field of online privacy. In particular, we have made great strides in the areas of remote execution, distributed computing, cryptography, and differential privacy - but our documentation has been a source of confusion and discouragement for our community members. The basic examples and tutorials illustrate the use of PySyft, but the examples are quickly broken by the fast pace of development. This ends up burying our development teams in support requests that are hard to keep up with.

[OpenMined’s GSoD Project Ideas page](https://www.notion.so/1c65e1bbacfe4bc2b6a565445b4ec57f)

We are requesting additional support in an effort to improve the inline documentation of PySyft, as well as producing public API references, and general readme documentation.

We aim to prevent the blurring that often occurs among various documentation types: Tutorials, How-To Guides, Discussions, and References in order to keep our technical debt in check.

### Our Projects Scope

The scope of the PySyft project will:

- Audit the existing documentation and create a friction log of the current state of the PySyft documentation for our top use cases
  - Audit findings will be published
  - Actionable issues list will be created that address these gaps
  - Actionable issues list will be made easily accessible to team members and the wider community so that they can be actioned
- API references - improve inline documentation with concise explanations of key concepts and examples
- General documentation - identify issues in more detail pertaining to inline documentation, general README documentation
  - Add reference material to describe interactions with other OpenMined technologies. Concise with code based structure
  - How-to guides to solve specific problems
- Both use-case tutorials and examples:
  - Update core use-cases as they pertain to the PySyft library
  - Add to showcase various privacy preserving use-cases for canonical tasks (e.g. facial recognition, credit defaulting)
  - Create supplemental Jupyter notebooks that highlight those use-cases
- Build a standardized framework for future OpenMined library documentation needs

### Out of Scope Work

- Other OpenMined libraries which have similar documentation needs
- Generalized GitHub tutorials; instead, a list of Actionable issues will link to material that is targeted and relevant
- Discussion forum creation because we are already very active with our community in Slack and have specific channels for discussing topics

### Docs Team

OpenMined has an existing documentation team of 15 people from our community, and is led by two part-time individuals. We would prefer to utilize funding on members within our existing community and documentation team, but would also be open to new members joining our documentation team during the lifespan of Google Season of Docs. For reference, we have done a nearly identical setup previously for Google Summer of Code in 2020, and in 2021. Nearly everyone in our community is a volunteer, only having a small crew of dedicated full-time staff that are funded by other means. Our documentation team leads have dedicated themselves to seeing through the entire Google Season of Docs project, in addition to their team leadership responsibilities.

### Measuring our Project's Success

The Documentation Team at OpenMined is new, and we have plans to begin a full-sweep documentation audit of PySyft with the funds received from Google Season of Docs. The Documentation Team is dedicated to releasing their audit findings publicly and will subsequently be creating a variety of issues on those repositories for team members and our broader open-source community to action. Any Google Season of Docs member is included on the team for the duration of their membership and would likely be invited to join the team after the project is officially over. Our hope is that this improved documentation will result in targeted pull requests, pull requests from new contributors, and more pull requests overall.

Success metrics are loosely defined at the moment, as the team is new to our organization. Generally speaking, metrics will likely look like the following:

- Audit based on four areas of documentation:
  - Examples and tutorials
  - General documentation
  - API references
  - Inline documentation (error code documentation, style guide information, etc.)
- Once an audit is received by the managing team and by the Engineering Lead, issues will be created based upon the findings of the audit. It’s expected that the number of issues will range from 40-100 issues that are specific only to documentation (no code related issues or bug reports).
- From here, the Documentation Team will have roughly 2 months to close at least 80% of the issues and allow 20% or so to be handled by our broader community. Allowing the community the opportunity to close documentation issues serves as a recruitment tool for the Documentation Team.
- Increases in documentation pull requests where measurement begins the quarter after publication:
  - Actionable issue pull requests → target 15%, stretch 20%
  - New contributor pull requests → target 10%, stretch 15%
  - Repeat contributor pull requests → target 5%, stretch 10%
  - Internal response time from PR to Merge for small, medium PRs. → target 2 weeks, stretch 1 week. Exceptions for large PRs, flag PRs over 1 month old.
  - Count visits to error code documentation vs. error code issues opened → measure weekly to monitor
  - Increased number of forks and stars of the PySyft repository →target 10%

### Timeline

The PySyft Documentation project will take approximately 6 months to complete with 1-2 technical writers. We will start with our project kick off and technical writer orientation which will remain ongoing throughout the GSoD program. We’d like to get a head start on the audit phase in May extending through July where the outcome will be an actionable issues list to share with the community. The next 3 months will be spent creating PySyft documentation that has been sequenced and prioritized for greatest impact.

### Dates & Activities

14 April - GSoD application decisions

15 April - Hire up until May 16 deadline. Begin kick off and orientation phase asap

15 May - Ongoing orientation & Audit phase start

June-July - Audit existing documentation and create an Actionable Issues list

Evaluation #1 due June 15-22

Evaluation #2 due July 15-22

August 15 - Transition from Audit to Creation phase. Publish Actionable issues list

Evaluation #3 due Aug 15-22

Sept-Oct - Create documentation & publishing activities

Evaluation #4 due Sept 15-22

Evaluation #5 due Oct 15-22

November - Project completion

Final Evaluation due Nov 15-30

Case study summary

Metric follow up and document for next year comparing Q2(AMJ)2022 to Q4(OND)2022, Q3(JAS)2022 to Q1(JFM)2023

### Project budget

<table>
  <tr>
   <td><strong>Budget item</strong>
   </td>
   <td><strong>Amount</strong>
   </td>
   <td><strong>Running Total</strong>
   </td>
   <td><strong>Notes/justifications</strong>
   </td>
  </tr>
  <tr>
   <td>1-2 technical writers for PySyft 
<p>
(Audit phase - 3 mths)
   </td>
   <td>6500
   </td>
   <td>6500
   </td>
   <td>40-100 issues identified, listed, prioritized & made actionable
   </td>
  </tr>
  <tr>
   <td>1-2 technical writers for PySyft (Creation phase - 3 mths)
   </td>
   <td>6500
   </td>
   <td>13000
   </td>
   <td> Updates are sequenced, additions are prioritized, actioned and published.
   </td>
  </tr>
  <tr>
   <td>PySyft + GSoD Project t-shirts (20 t-shirts)
<p>
(Team building)
   </td>
   <td>600
   </td>
   <td>13600
   </td>
   <td>T-shirts for our technical writers from GSoD,  team leads, and our community/docs team (given based on # of closed documentation issues)
   </td>
  </tr>
  <tr>
   <td>3 project leads PySyft
<p>
(Administration/Mentors/Metrics)
   </td>
   <td>1200
   </td>
   <td>14800
   </td>
   <td>3 volunteer stipends x 400 each. Documentation team leads/volunteers, who are not currently funded by OpenMined otherwise
   </td>
  </tr>
  <tr>
   <td>TOTAL
   </td>
   <td>
   </td>
   <td>14800
   </td>
   <td>
   </td>
  </tr>
</table>

### Additional Information

This will be the first time OpenMined participates in Google Season of Docs. We are now equipped to handle an influx of technical writers since we have a newly-formed Documentation Team to properly handle the additional managerial workload.

However, as mentioned elsewhere in the application, we’ve participated in Google Summer of Code twice previously. OpenMined’s leader (Andrew Trask @iamtrask) is a researcher at DeepMind, a company of Google/Alphabet.

GSoC has been one of the biggest drivers of participation in our community and is single-handedly responsible for hundreds of people joining our ever-growing community. Generally speaking, it’s understood that this proposal is better suited for open-source organizations that have 1 specific project in mind. To this end we are focusing our efforts to set the standard for our future documentation framework while expanding and bringing the documentation of PySyft up to date.

OpenMined has over 15 active repositories related to privacy-preserving machine learning and it is challenging to determine one and only one project to be served by Google Season of Docs. Our community will be thrilled with any amount of funding from GSoD for any number of technical writers. Given the size of our community and the number of projects we’re approaching, we believe that our proposal is fair, justified, and balanced in workload.
