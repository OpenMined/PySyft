import { UUID } from './uid.js';
import { v4 as uuidv4 } from 'uuid';

export class SyftMessage {
	address: UUID;
	id: UUID;
	reply_to: UUID;
	reply: boolean;
	kwargs: any;
	fqn: string;

	constructor(address: string, reply_to: string, reply: boolean, kwargs: any, fqn: string) {
		this.address = new UUID(address);
		this.id = new UUID(uuidv4());
		this.reply_to = new UUID(reply_to);
		this.reply = reply;
		this.kwargs = new Map(Object.entries(kwargs));
		this.fqn = fqn;
	}
}

export class SyftMessageWithReply extends SyftMessage {
	constructor(address: string, reply_to: string, kwargs: any, fqn: string) {
		super(address, reply_to, true, kwargs, fqn);
	}
}

export class SyftMessageWithoutReply extends SyftMessage {
	constructor(address: string, kwargs: string, fqn: string) {
		super(address, address, false, kwargs, fqn);
	}
}
