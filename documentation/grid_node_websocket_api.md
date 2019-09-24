# Grid Node Websocket API

All SocketIO endpoints will be detailed in this document.

## Connect

**Event** : `/connect`  
**Description** : Connect with grid node.  
**Payload** : Not needed.

## Set Node ID

**Event** : `/connect-node`  
**Description** : Connect two different nodes directly.  
**Payload-Type** : JSON  
**Auth required** : NO (can be changed) 

#### Event Payload
``` json
{
    "id" : "node_id",
    "uri" : "http://node_uri.com"
}
```

## Forward PySyft commands

**Event** : `/cmd`  
**Description**:  Forward pysyft command to hook virtual worker.  
**Payload-Type** : Syft Command (**Binary**)  
**Auth required** : NO (can be changed)
