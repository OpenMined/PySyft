from .scope import Scope


class ScopeController:
    """ This class implements controller design pattern over the scopes."""

    def __init__(self):
        self.scopes = {}

    def create_scope(self, worker_id: str, protocol):
        """ Register a new scope
            
            Args:
                worker_id : id used to identify the creator of this scope.
                protocol :  Protocol structure used by this scope.
            Returns:
                scope : Scope Instance.
        """
        scope = Scope(worker_id, protocol)
        self.scopes[scope.id] = scope
        return self.scopes[scope.id]

    def delete_scope(self, scope_id):
        """ Remove a registered scope.

            Args:
                scope_id : Id used identify the desired scope. 
        """
        del self.scopes[scope_id]

    def get_scope(self, scope_id):
        """ Retrieve the desired scope.
            
            Args:
                scope_id : Id used to identify the desired scope.
            Returns:
                scope : Scope Instance or None if it wasn't found.
        """
        return self.scopes.get(scope_id, None)
