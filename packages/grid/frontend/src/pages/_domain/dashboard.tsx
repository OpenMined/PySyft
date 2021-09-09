import {useMemo} from 'react'
import {entityColors} from '@/utils'
import {DashboardCards, LatestAssetsList} from '@/components/pages/dashboard'
import {Page} from '@/components'
import {useDatasets, useRequests, useModels, useTensors, useUsers} from '@/lib/data'
import {AdjustmentsIcon, BookOpenIcon, DatabaseIcon, UsersIcon} from '@heroicons/react/outline'

export default function Dashboard() {
  // FIX: Dashboard route is not working...
  const {all: datasetsAll} = useDatasets()
  const {all: requestsAll} = useRequests()
  const {all: modelsAll} = useModels()
  const {all: tensorsAll} = useTensors()
  const {all: usersAll} = useUsers()
  const {data: datasets} = datasetsAll()
  const {data: requests} = requestsAll()
  const {data: models} = modelsAll()
  const {data: tensors} = tensorsAll()
  const {data: users} = usersAll()

  const latestAdditions = useMemo(() => ({datasets, models, tensors}), [datasets, models, tensors])

  return (
    <Page title="Dashboard" description="Grid Domain statistics">
      <div className="space-y-12">
        <section>
          <div>
            <DashboardCards
              cards={[
                {
                  link: '/users',
                  bgColor: entityColors.users,
                  icon: UsersIcon,
                  main: 'Total registered users',
                  value: users?.length
                },
                {
                  link: '/requests',
                  bgColor: entityColors.requests,
                  icon: BookOpenIcon,
                  main: 'Pending data requests',
                  value: requests?.length
                },
                {
                  link: '/datasets',
                  bgColor: entityColors.datasets,
                  icon: DatabaseIcon,
                  main: 'Datasets available',
                  value: datasets?.length
                },
                {
                  link: '/models',
                  bgColor: entityColors.models,
                  icon: AdjustmentsIcon,
                  main: 'Models available',
                  value: models?.length
                }
              ]}
            />
          </div>
        </section>
        <section className="space-y-2">
          <h2 className="text-xl font-medium">Latest Domains, Models and Tensors</h2>
          <div className="overflow-hidden border border-gray-200 rounded-md">
            {/* TODO: search box? filters? */}
            <LatestAssetsList {...latestAdditions} />
          </div>
        </section>
      </div>
    </Page>
  )
}
