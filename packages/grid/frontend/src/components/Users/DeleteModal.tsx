import { faExclamationTriangle } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Button, H4, Text } from '@/omui'
import Modal from '@/components/Modal'
import { useUsers } from '@/lib/data'

function DeleteModal({ show, userIds, onClose }) {
  return (
    <Modal show={show} onClose={onClose}>
      <div className="flex w-full justify-center">
        <FontAwesomeIcon
          icon={faExclamationTriangle}
          className="text-warning-500"
        />
        <H4>Are you sure you want to delete the selected users</H4>
        <Text>
          If deleted the selected user(s) will have their accounts removed from
          your domain node and will have all pending requests closed.
        </Text>
        <div className="flex space-x-3">
          <Button
            variant="primary"
            className="bg-error-500"
            size="sm"
            onClick={() => trigger(userIds)}
          >
            Delete User
          </Button>
          <Button variant="ghost" type="button" onClick={onClose} size="sm">
            Cancel
          </Button>
        </div>
      </div>
    </Modal>
  )
}

function trigger(ids: Array<string>) {
  ids.forEach((id) => useDeleteUser({ id }))
}

function useDeleteUser({ id }) {
  const delUser = useUsers().remove(id).mutate
  delUser()
}

export { DeleteModal }
