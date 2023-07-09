import string
import random

def password_generator(length):
    """ Function that generates a password given a length """

    uppercase_loc = random.randint(1,4)  # random location of lowercase
    symbol_loc = random.randint(5, 6)  # random location of symbols
    lowercase_loc = random.randint(7,12)  # random location of uppercase

    password = ''  # empty string for password

    pool = string.ascii_letters + string.punctuation  # the selection of characters used

    for i in range(length):

        if i == uppercase_loc:   # this is to ensure there is at least one uppercase
            password += random.choice(string.ascii_uppercase)

        elif i == lowercase_loc:  # this is to ensure there is at least one uppercase
            password += random.choice(string.ascii_lowercase)

        elif i == symbol_loc:  # this is to ensure there is at least one symbol
            password += random.choice(string.punctuation)

        else:  # adds a random character from pool
            password += random.choice(pool)

    return password  # returns the string

class Credential():
    
    def __init__(self, name, email, password, ip, port, domain_name=None, domain_id=None):
        self.name = name
        self.email = email
        self.password = password
        self.ip = ip
        self.port = str(port)
        self.domain_name = domain_name
        self.domain_id = str(domain_id)
        
    @property
    def onepass_title(self):
        if self.domain_name is not None:
            return self.domain_name + ":" + self.domain_id
        else:
            return "Main Syft Credentials"
    
    @property
    def onepass_username(self):
        return self.name +":" + self.email
        
    @property
    def onepass_url(self):
        return self.ip+":"+self.port
        
    # it doesn't seem to include the password when listing all items in a vault
    # such as by calling op.item_list(vault="Syft")
    @staticmethod
    def from_high_level_1password_dict(d):
        if d['title'] == 'Main Syft Credentials':
            
            name_and_email = d['additional_information']
            name, email = name_and_email.split(":")
            password = ""
            url = d['urls'][0]['href']
            
            return Credential(name=name, email=email, password=password, ip=url, port=80)
        
        else:
            
            name, email = d['additional_information'].split(":")
            domain_name, domain_id = d['title'].split(":")
            password = ""
            ip, port = d['urls'][0]['href'].split(":")
            return Credential(name=name, 
                              email=email, 
                              password=password, 
                              ip=ip,
                              port=port,
                              domain_name=domain_name,
                              domain_id=domain_id)
        
    @staticmethod
    def from_full_1password_dict(login_creds):
        if login_creds['title'] != "Main Syft Credentials":
            domain_name, domain_id = login_creds['title'].split(":")
            ip, port = login_creds['urls'][0]['href'].split(":")            
        else:
            domain_name = None
            domain_id = None
            ip = login_creds['urls'][0]['href'].split(":")[0]
            port = login_creds['urls'][0]['href'].split(":")[1]

        name_and_email = list(filter(lambda x:x['id'] == 'username', login_creds['fields']))[0]['value']
        name = name_and_email.split(":")[0]
        email = name_and_email.split(":")[1]        
        password = list(filter(lambda x:x['id'] == 'password', login_creds['fields']))[0]['value']
        return Credential(name=name, 
                          ip=ip,
                          port=port,
                          email=email, 
                          password=password, 
                          domain_name=domain_name, 
                          domain_id=domain_id)
    
    


class Wallet:
    
    def __init__(self):
        print("Syft Wallet is experimental and it requires:")
        print("\t- 1Password for Mac: https://1password.com/product/mac/")
        print("\t- A vault in 1Password called 'Syft'")
        print("\t- A python package: https://github.com/zcutlip/pyonepassword/tree/main")
        print("\t- A command line tool: https://developer.1password.com/docs/cli/get-started/")

        self.op = self.do_signin()
        self.check_syft_vault_exists_or_error()
        self.check_main_syft_credentials()

    @property
    def email(self):
        # login_creds = op.item_get(vault="Syft", item_identifier=domain_name)
        # list(filter(lambda x:x['id'] == 'username', login_creds['fields']))[0]['value']
        return self.op.item_get(vault="Syft", item_identifier="Main Syft Credentials")['fields'][0]['value'].split(":")[-1]
    
    @property
    def name(self):
        # login_creds = op.item_get(vault="Syft", item_identifier=domain_name)
        # list(filter(lambda x:x['id'] == 'username', login_creds['fields']))[0]['value']
        return self.op.item_get(vault="Syft", item_identifier="Main Syft Credentials")['fields'][0]['value'].split(":")[0]    
    
    def check_syft_vault_exists_or_error(self):
        if "Syft" not in self.vaults:
            raise Exception("'Syft' vault not found. Please go to 1password and create a vault called 'Syft'")
        
    def check_main_syft_credentials(self):
        syft_vault = self.op.item_list(vault="Syft")
        if 'Main Syft Credentials' not in set(map(lambda x:x['title'], syft_vault)):
            print("WELCOME!")
            print("It looks like this is the first time you've used your Syft Wallet")
            print("Please specify the email you'd like to use to login to PySyft Domains")
            email = input()
            print("And what is your name?")            
            name = input()
            print("And what is your website url?")
            url = input()
            print("Thank you!")
            
            cred = Credential(email=email, name=name, ip=url, port=80, password="")
            
            wallet.create_login(cred) # just putting this here for now
                  
        
    @property
    def vaults(self):
        return set(map(lambda x:x['name'], self.op.vault_list()))
        
    # See examples/example-sign-in.py for more sign-in examples
    def do_signin(self):

        import getpass

        from pyonepassword import OP
        from pyonepassword.api.exceptions import (
            OPSigninException,
            OPItemGetException,
            OPNotFoundException,
            OPConfigNotFoundException
        )    

            # Let's check If biometric is enabled
        # If so, no need to provide a password
        if OP.uses_biometric():
            try:
                # no need to provide any authentication parameters if biometric is enabled
                op = OP()
            except OPAuthenticationException:
                print("Uh oh! Sign-in failed")
                exit(-1)
        else:
            # prompt user for a password (or get it some other way)
            my_password = getpass.getpass(prompt="1Password master password:\n")
            # You may optionally provide an account shorthand if you used a custom one during initial sign-in
            # shorthand = "arbitrary_account_shorthand"
            # return OP(account_shorthand=shorthand, password=my_password)
            # Or we'll try to look up account shorthand from your latest sign-in in op's config file
            op = OP(password=my_password)
        return op
    

    def create_login(self, cred):

        title = cred.onepass_title
        username = cred.onepass_username
        url = cred.onepass_url
        vault = "Syft"
            
        import os
        try:
            new_item: OPLoginItem = self.op.login_item_create(title,
                                                         username,
                                                         url=url,
                                                         password=cred.password,
                                                         vault=vault)
        except Exception as e:
            print(e)
            # workaround for https://1password.community/discussion/140523/cannot-create-an-item-from-template-and-stdin-at-the-same-time
            new_cmd = "cat " + e.msg[-3] + " | "
            new_cmd += "op item create --vault " + e.msg[-1]
            new_cmd += " > /dev/null 2>&1"
            out = os.system(new_cmd)    
    
    def login(self, guest_domain):
        domain = guest_domain
        domain_id = str(domain.id)

        syft_vault = self.op.item_list(vault="Syft")
        syft_vault = self.credentials
        if domain_id not in set(map(lambda x:x.domain_id, syft_vault)):
        
            new_cred = Credential(name=wallet.name,
                                  email=wallet.email,
                                  password=password_generator(20),
                                  ip=domain.route.host_or_ip,
                                  port=domain.route.port,
                                  domain_name=domain.name,
                                  domain_id=str(domain.id))

            wallet.create_login(new_cred)

        cred = list(filter(lambda x:x.domain_id == domain_id, self.credentials))[0]

        result = domain.register(name=cred.name,
                        email=cred.email,
                        password=cred.password)

        new_domain = domain.login(email=cred.email, password=cred.password)  
        
        if new_domain.logged_in_user == '':
            raise Exception("Looks like this user already exists but "+\
                            "your wallet has the wrong password. Contact the domain owner for help.")
        
        return new_domain
    
    @property
    def credentials_no_password(self):
        syft_vault = self.op.item_list(vault="Syft")
        creds = list()
        for c in syft_vault:
            creds.append(Credential.from_high_level_1password_dict(c))
        return creds
    
    @property
    def credentials(self):
        creds = list()
        for c in self.credentials_no_password:
            d = self.op.item_get(vault="Syft", item_identifier=c.onepass_title)
            creds.append(Credential.from_full_1password_dict(d))
        return creds

    @property
    def domains(self):
        import syft as sy
        domains = list()
        for cred in self.credentials:
            if cred.domain_name is not None:
                domain = sy.login(email=cred.email, password=cred.password, url="http://"+cred.ip, port=cred.port)
                domains.append(domain)
        return domains
                
        