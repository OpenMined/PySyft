import { useState } from 'react'
import Link from 'next/link'
import { faCheck } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Button, Input, FormControl, Text } from '@/omui'
import Modal from '../Modal'
import { useUsers } from '@/lib/data'

export function PrivacyBudgetModal({ show, onClose, user }) {
  const { mutate: update, isLoading } = useUsers().update(user?.id, {
    onSuccess: onClose,
  })

  const [value, setValue] = useState(user?.budget)
  // TODO: mitigated for demo
  const add = () => setValue((v) => (Number(v) + 0.1).toFixed(2))
  const subtract = () => setValue((v) => (Number(v) - 0.1).toFixed(2))
  const upgrade = () => update({ budget: Number(value) })

  return (
    <Modal show={show} onClose={onClose} className="max-w-3xl">
      <div className="col-span-full">
        <FontAwesomeIcon icon={faCheck} className="font-bold text-3xl" />
        <Text as="h1" className="mt-3" size="3xl">
          Upgrade Budget
        </Text>
        <Text className="mt-4 whitespace-pre-wrap" as="p">
          {`Allocating Privacy Budget (PB) is an optional setting that allows you to maintain a set standard of privacy while offloading the work of manually approving every data request for a single user. You can think of privacy budget as credits you give to a user to perform computations from. These credits of Epsilon(ɛ) indicate the amount of visibility a user has into any one entity of your data. The more budget the more visibility. By default all users start with 0ɛ and must have their data requests approved manually until upgraded.\n\nYou can learn more about privacy budgets and how to allocate them at `}
          <Link href="https://courses.openmined.org">
            <a className="text-primary-500">Course.OpenMined.org</a>
          </Link>
          .
        </Text>
      </div>
      <div className="col-span-4 mt-2.5">
        <FormControl label="Adjust Privacy Budget" id="role" className="mt-6">
          <Input
            type="number"
            min={user?.budget_spent}
            step={0.1}
            addonRight="+"
            addonLeft="-"
            addonLeftProps={{ onClick: subtract }}
            addonRightProps={{ onClick: add }}
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
        </FormControl>
      </div>
      <div className="col-span-full flex justify-between mt-12">
        <Button
          size="sm"
          variant="outline"
          onClick={onClose}
          disabled={isLoading}
        >
          Cancel
        </Button>
        <Button
          size="sm"
          variant="primary"
          onClick={upgrade}
          isLoading={isLoading}
        >
          Upgrade
        </Button>
      </div>
    </Modal>
  )
}
