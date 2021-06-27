# FAQ 0.2.x ➡️ 0.3.x

## Got a Question?
If you have a question which isn't listed here, Open a PR or come into Slack and we can try to answer it as best as we can. 😊

## Where is the 0.2.x code?
The code can be found on a branch called [0.2.x](https://github.com/OpenMined/PySyft/tree/syft_0.2.x)

## Will you support 0.2.x any more?
We will help merge suitable PRs to 0.2.x branch until such time as 0.3.x completely replaces the functionality in 0.2.x.

## Why are they not compatible?
The way 0.2.x was implemented was not viable long term so we were forced to rewrite in a way which makes the underlying code significantly different.

## When will Protocols be available?
We plan on supporting protocols and other forms of Plans very soon however due to internal milestones which are already in development we can't provide a firm date as yet.

## Where is Feature X?
While the goal is to reach feature parity and significantly surpass it, please bear with us as we have a lot of work to support some crucial features needed for our own internal roadmap.

## What is Duet and how is it different to Syft?
Duet is an opt-in part of the Syft code base that is dedicated to providing quick and easy peer-to-peer data science over Syft. For data experiments and initial collaboration between parties Duet is a great way to start using Syft and the OpenMined ecosystem of PPML tools.

## Can I run Duet outside of a Notebook
Technically yes, although there can be some issues with the code expecting some packages from jupyter in some places. Longer term we have a different suite of tools which are more appropriate for CLI usage. As Duet continues to evolve you should expect more and more beneficial features based around notebook style UI. This will not prevent you from using Syft without notebooks.

## How do I do MLOps with Syft and Duet or connect more than two systems?
Anything more automated or scaleable past p2p will be handled by a separate parallel project called [Grid](https://github.com/OpenMined/pygrid). The two teams work very closely and there are some areas of overlap so rest assured the Syft team is well aware of the desire for more granular user permission systems, many to many connections and cloud deployment / orchestration and they are coming!

## Why am I getting ModuleNotFoundError?
This is probably because you are using the latest version of PySyft.
If you need PySyft 0.2.x you can install it by choosing a specific version such as: `pip install syft==0.2.9`.

**Note** If you are pursuing the [Udacity - Secure and Private AI Course](https://www.udacity.com/course/secure-and-private-ai--ud185), you need version `0.1.2a1`. Simply run: `pip install syft==0.1.2a1`
