import { UUID } from './uid.js'
import { v4 as uuidv4 } from 'uuid';

export class SyftMessage {

    address: UUID;
    id: UUID;
    reply_to: UUID;
    reply: Boolean;
    kwargs: any;
    fqn: String;

    constructor(address: String, reply_to: String, reply: Boolean, kwargs: any, fqn: String) {
        this.address = new UUID(address);
        this.id = new UUID(uuidv4());
        this.reply_to = new UUID(reply_to);
        this.reply = reply
        this.kwargs = new Map(Object.entries(kwargs))
        this.fqn = fqn
    }
}

export class SyftMessageWithReply extends SyftMessage {
    constructor(address: String, reply_to: String, kwargs: any, fqn: String){
        super(address,reply_to,true, kwargs, fqn)
    }
}

export class SyftMessageWithoutReply extends SyftMessage {
    constructor(address: String, kwargs: String, fqn: String){
        super(address,address,false, kwargs, fqn)
    }
}