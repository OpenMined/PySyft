import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCheck, faTimes } from '@fortawesome/free-solid-svg-icons'

function AcceptDeny({ onAccept, onDeny }) {
  return (
    <div className="space-x-3 flex-shrink-0 flex text-white">
      <button
        className="rounded-full bg-gray-200 w-7 h-7 flex justify-center items-center hover:bg-primary-500"
        onClick={onAccept}
      >
        <FontAwesomeIcon icon={faCheck} />
      </button>
      <button
        className="rounded-full bg-gray-200 w-7 h-7 flex justify-center items-center hover:bg-primary-500"
        onClick={onDeny}
      >
        <FontAwesomeIcon icon={faTimes} />
      </button>
    </div>
  )
}

export { AcceptDeny }
