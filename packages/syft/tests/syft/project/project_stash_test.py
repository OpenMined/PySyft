# third party
from faker import Faker
from syft.core.node.new.document_store import PartitionKey
from syft.core.node.new.project_stash import ProjectStash

from syft import UserCodeStatus
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.project import ProjectSubmit, Project
from syft.core.node.new.syft_object import SyftObject
from syft.core.node.new.context import AuthedServiceContext

class MockSyftObject(SyftObject):
  __canonical_name__ = "MockSyftObject"
  __version__ = 1
  
  name: str

def create_mock_project(faker, authed_context: AuthedServiceContext):

   mock_obj = MockSyftObject(name=faker.name())  # add a mock name
   
   # refer to project.py to see the argument for ProjectSubmit and Project. These classes inherit from SyftObject and are pydantic classes.
   # NOTE: We can convert one SyftObject to another via transform decorators and via the `to` method in SyftObject class
   # you will see one for @transform(ProjectSubmit, Project) in project.py, it means that when ProjectSubmit is converted to Project
   # the list of functions defined will be called iteratively on the ProjectSubmit object to finally convert it to Project.

   mock_project_submit_obj = ProjectSubmit(
     name="", # add a dummy name
     description=""  # add a dummy description, you can use faker for generating a random text
   )   
   mock_project_submit_obj.add_request(obj=mock_obj, permission=UserCodeStatus.EXECUTE)
   mock_project = mock_project_submit_obj.to(Project, context=authed_context)
   return mock_project



def save_mock_project(project_stash, project):
   # prepare: add mock data to database
    result = project_stash.partition.set(project)
    assert result.is_ok()
    saved_project = result.ok()
    return saved_project


ProjectUserVerifyKeyPartitionKey = PartitionKey(
    key="user_verify_key", type_=SyftVerifyKey
)

def test_projectstash_get_all_for_verify_key(project_stash: ProjectStash,verify_key: ProjectUserVerifyKeyPartitionKey) -> None:

  #create atuhedcontext class and pass the verify key
   auth_context = AuthedServiceContext(credentials = verify_key)

   mock_project = create_mock_project(Faker, auth_context )
   # call method to add mock project to database
   saved_mock_project = save_mock_project(project_stash,mock_project) #???

   # create you verify key partition key

   # tests the method
   # check if results are is_ok
   assert saved_mock_project.is_ok()
   result = project_stash.get_all_for_verify_key(verify_key)
   assert result.is_ok()
   # check if the retrieved project is equivalent to the project saved.
   assert result == saved_mock_project
   # another case to test could be if you pass a random verify key.

