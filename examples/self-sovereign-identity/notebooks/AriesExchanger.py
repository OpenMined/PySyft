#!/usr/bin/env python
# coding: utf-8

# In[1]:


import syft as sy
import time


# In[2]:


from syft.grid.duet.exchange_ids import DuetCredentialExchanger
from typing import Any as TypeAny
import json


# In[3]:


get_ipython().run_line_magic('autoawait', '')
import time
import asyncio
import nest_asyncio
from aries_basic_controller.aries_controller import AriesAgentController

    



class AriesCredentialExchanger(DuetCredentialExchanger):
    def __init__(self, agent_controller: TypeAny) -> None:
        super().__init__()
        self.agent_controller = agent_controller
        self.responder_id = None
        self.register_agent_listeners()

    def run(
        self,
        credential: str,
    ) -> str:
        self.credential=credential
        if self.join:
            self._client_exchange(duet_token = credential)
        else: 
            self._server_exchange(duet_token = credential)
           
        while True:
            time.sleep(2)
            print("Current Responder ID", self.responder_id)
            if self.responder_id:
                break
                

        return self.responder_id
    
    def _client_exchange(self, duet_token):
        
        while True:
            invite = input("♫♫♫ > Duet Partner's Aries Invitation: ")  # nosec
            loop = asyncio.get_event_loop()
            is_ready = False
            try: 
                response = loop.run_until_complete(self.agent_controller.connections.accept_connection(invite))
                print(response["connection_id"])
                connection_id = response["connection_id"]            
            except:
                print("    > Error: Invalid invitation. Please try again.")
            break

        
        print("Waiting for active connection", connection_id)
        self.await_active(connection_id)

               
        time.sleep(1)

        print("Sending Duet Token", connection_id, duet_token)

        
        loop.run_until_complete(self.agent_controller.messaging.send_message(connection_id, duet_token))
        return

            
                
        
    def _server_exchange(self, duet_token):
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(self.agent_controller.connections.create_invitation())
        connection_id = response["connection_id"]
        invite_message = json.dumps(response['invitation'])

        print()
        print(
            "♫♫♫ > "

            + "STEP 1:"
            + " Send the aries invitation to your Duet Partner!"
        )
        print()
        print(invite_message)
        print()
        print("Waiting for active connection", connection_id)
        self.await_active(connection_id)

        time.sleep(1)
        print("Sending Duet Token", connection_id, duet_token)
        loop.run_until_complete(self.agent_controller.messaging.send_message(connection_id, duet_token))
        return

    def await_active(self, connection_id):
        while True:
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(self.agent_controller.connections.get_connection(connection_id))
            is_ready = "active" == response["state"]
            if is_ready:
                print("Connection Active")
                break
            else: 
                time.sleep(2)
        return

    
    def register_agent_listeners(self):
        print("REGISTER LISTENERS")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.agent_controller.listen_webhooks())
        
        listeners = [{
            "handler": self.messages_handler,
            "topic": "basicmessages"
        },{
            "topic": "issue_credential",
            "handler": self.cred_handler
        },{
            "handler": self.connection_handler,
            "topic": "connections"
        },{
            "topic": "present_proof",
            "handler": self.proof_handler
        }]
        self.agent_controller.register_listeners(listeners, defaults=True)
        return
        
    def cred_handler(self, payload):
        print("Handle Credentials")
        exchange_id = payload['credential_exchange_id']
        state = payload['state']
        role = payload['role']
        attributes = payload['credential_proposal_dict']['credential_proposal']['attributes']
        print(f"Credential exchange {exchange_id}, role: {role}, state: {state}")
        print(f"Offering: {attributes}")
    
    def connection_handler(self, payload):
        print("Connection Handler Called")
        connection_id = payload["connection_id"]
        state = payload["state"]
        print(f"Connection {connection_id} in State {state}")
        
        
    def proof_handler(self, payload):
        print("Handle present proof")
        role = payload["role"]
        pres_ex_id = payload["presentation_exchange_id"]
        state = payload["state"]
        print(f"Role {role}, Exchange {pres_ex_id} in state {state}")
        
    # Receive basic messages
    def messages_handler(self, payload):
        print("Handle Duet ID", payload["content"])    
        self.responder_id = payload["content"]
        
    ## Used for other Aries connections. E.g. with an issuer
    def accept_connection(self, invite):
        # Receive Invitation
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(self.agent_controller.connections.accept_connection(invitation))
        # Print out accepted Invite and Alice's connection ID
        print("Connection", response)
        return response["connection_id"]
        
    def create_invitation(self, is_duet: bool = False):
        # Create Invitation
        loop = asyncio.get_event_loop()
        invite = loop.run_until_complete(self.agent_controller.connections.create_invitation())
        connection_id = invite["connection_id"]
        invite_message = json.dumps(invite['invitation'])
#         print("Connection ID", dataowner_id)
#         print("Copy Invitation to Data Owner\n")
        return invite_message

    