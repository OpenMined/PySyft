import { List, ListIconItem, ListItem, ListInnerContainer } from '@/omui'
import { AcademicCapIcon, BellIcon, QuestionMarkCircleIcon } from '@heroicons/react/solid'
import { CircleStatus } from './DomainStatus'

const QuickNav = () => {
  return (
    <List className="flex gray-600" horizontal>
      <ListIconItem icon={AcademicCapIcon} />
      <ListIconItem icon={QuestionMarkCircleIcon} />
      <ListItem>
        <ListInnerContainer>
          <CircleStatus />
        </ListInnerContainer>
      </ListItem>
      <ListIconItem icon={BellIcon} />
    </List>
  )
}

export { QuickNav }
