# grid absolute
from app.models.item import Item
from app.schemas.item import ItemCreate
from app.schemas.item import ItemUpdate

# relative
from .base import CRUDBase
from .crud_item import item
from .crud_user import user

# For a new basic set of CRUD operations you could just do


item = CRUDBase[Item, ItemCreate, ItemUpdate](Item)
