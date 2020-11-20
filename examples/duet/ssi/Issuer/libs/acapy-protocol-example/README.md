# Aries ACA-Py Protocol Example

This repository contains a basic example of a Hyperledger Aries protocol implemented in python and included into the [ACA-Py cloud-agent](https://github.com/hyperledger/aries-cloudagent-python) as a plugin.

Hyperledger Aries is a project enabling trusted online peer-to-peer interactions through a set of protocols outlined in [aries-rfcs](https://github.com/hyperledger/aries-rfcs). 

If you are new to Hyperledger Aries I recommend working through these two Edx courses:
* [Introduction to Hyperledger Sovereign Identity Blockchain Solutions: Indy, Aries & Ursa](https://www.edx.org/course/identity-in-hyperledger-aries-indy-and-ursa)
* [Becoming a Hyperledger Aries Developer](https://www.edx.org/course/becoming-a-hyperledger-aries-developer)

This is a basic example of a protocol implemented for an aca-py agent and included as a plugin. It can be thought of as an extension to [Chapter 5](https://courses.edx.org/courses/course-v1:LinuxFoundationX+LFS173x+1T2020/course/#block-v1:LinuxFoundationX+LFS173x+1T2020+type@chapter+block@002f6693698443ceb77443a8d50cb974) of the Becoming a Hyperledger Aries Developer edx course.

I think it is worth reiterating some of the content from Chapter 5 - it took me a while to internalize this and appreciate what it means from a development perspective. Hyperledger Aries is a set of specifications for protocols expressed as [aries-rfcs](https://github.com/hyperledger/aries-rfcs). These specifications can be implemented in any language and enable aries agents to be spun up, much like a server (aca-py start instead of express().listen()). 

The primary use case for agents is secure, private peer-to-peer communication with other agents using a protocol called DIDComm. DIDComm defines how messages should be secured and sent to other agents, but does not describe the content of these messages. The majority of aries protocols are simply defining a set of messages the get communicated back and forth between agents. Hyperledger aries defines a set of protocols as [aries-rfcs](https://github.com/hyperledger/aries-rfcs/tree/master/features), aswell as an [interoperability profile](https://github.com/hyperledger/aries-rfcs/tree/master/concepts/0302-aries-interop-profile) defining the core set of protocols agents must implement if they wish to be interoperable. 

In addition to the protocol's already implemented in Hyperledger aries projects, it is possible to define protocols for specific usecases. As long as you describe in the code of the agent a set of **Messages**, each with their own unique **MessageType**, **MessageSchema** and **Handler**. This is so that when the agent recieves a **Message** of a particular **MessageType**, it understands the expected structure of the message and how to **Handle** it. Handling a **Message** can include sending a webhook to the **Controller** or replying to the sender with a new Message.

I have uploaded a walkthrough [video that dives into the ACA-Py implementation details of a protocol](https://www.youtube.com/watch?v=HjD-fasHmX8). 

To run through the example clone the repo then run `./manage up`. This starts 2 aca-py agents with the plugin protocolexample included and defined in acapy_protocol_example.protocolexample.

You can navigate to the swagger API for the two agents at:
* http://localhost:8021/api/doc
* http://localhost:8051/api/doc

Simply create a connection between the two agents using the swagger interface - follow [this demo](https://github.com/hyperledger/aries-cloudagent-python/blob/master/demo/AriesOpenAPIDemo.md) if you are unsure how. Then invoke the protocolexample protocol using the swagger interface to send a protocolexample message from one agent to the other. You should see this in the logs.
