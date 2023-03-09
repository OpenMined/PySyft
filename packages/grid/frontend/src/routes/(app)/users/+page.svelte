<script>
  import Badge from '$lib/components/Badge.svelte';
  import Button from '$lib/components/Button.svelte';

  import Fa from 'svelte-fa';
  import { onMount } from 'svelte';
  import { url } from '$lib/stores/nav';
  import { parseActiveRoute } from '$lib/helpers';
  import { faTrash, faCircleInfo } from '@fortawesome/free-solid-svg-icons';

  export let location;

  onMount(() => url.set(parseActiveRoute(location.pathname)));
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full">
  <div class="flex-col p-8">
    <div class="flex justify-between">
      <div class="flex items-center">
        <h2 class="pr-3 font-rubik text-2xl">Jane Doe</h2>
        <Badge variant="primary-light">Data Scientist</Badge>
      </div>
      <div class="flex justify-right items-center" style="color: #8F8AA8">
        <Button variant="ghost" option="right" onClick={() => alert('TODO: Delete User')}>
          <Fa icon={faTrash} size="xs" />
          <p class="pl-0.5 text-xs">Delete User</p>
        </Button>
      </div>
    </div>
    <a class="text-xs" href="/">Change Role</a>
  </div>

  <div class="space-y-4 px-8 mt-0 text-gray-600">
    <div class="font-roboto text-xs">
      <div class="flex">
        <h6 class="font-bold pb-2 pr-1">Privacy Budget</h6>
        <div style="color: #C7C4D4">
          <Fa icon={faCircleInfo} size="xs" />
        </div>
      </div>
      <div
        class="flex flex-row justify-between items-center border-solid border-2
        border-gray-100 rounded space-x-4 w-fit privacy-budget-background"
      >
        <div class="flex flex-col p-4">
          <div class="flex items-end">
            <span class="text-magenta-600 text-lg">10.00</span>
            <span class="text-magenta-600 text-lg pl-2">&#949;</span>
          </div>
          <span>Current Balance</span>
        </div>
        <div class="border-x border-gray-100 rounded h-8 mx-0" />
        <div class="flex flex-col py-4 pl-2">
          <div class="flex items-end">
            <span class="text-lg">10.00</span>
            <span class="text-lg pl-2">&#949;</span>
          </div>
          <span>Allocated Budget</span>
        </div>
        <div class="p-4">
          <Button
            variant="outlined-primary"
            option="margin-x-sm"
            onClick={() => alert('TODO: Adjust Budget')}>Adjust Budget</Button
          >
        </div>
      </div>
    </div>

    <div class="font-roboto text-xs">
      <h6 class="font-bold pb-2">Background</h6>
      <ul class="border-solid border-2 border-gray-100 rounded space-y-4">
        <li class="flex px-4 pt-4">
          <p class="font-bold pr-2">Email:</p>
          <a class="text-gray-600" href="mailto: jane.doe@abc.com">jane.doe@abc.com</a>
        </li>
        <li class="flex px-4">
          <p class="font-bold pr-2">Company/Institution:</p>
          <p>Oxford University</p>
        </li>
        <li class="flex px-4 pb-4">
          <p class="font-bold pr-2">Website/Profile:</p>
          <a class="text-gray-600" href="www.university.edu/research/profile/jane_doe"
            >www.university.edu/research/profile/jane_doe</a
          >
        </li>
      </ul>
    </div>

    <div class="font-roboto text-xs">
      <h6 class="font-bold pb-2">System</h6>
      <ul class="border-solid border-2 border-gray-100 rounded space-y-4">
        <li class="flex px-4 pt-4">
          <p class="font-bold pr-2">Date Added:</p>
          <p>2021-SEP-12 14:05</p>
        </li>
        <li class="flex px-4">
          <p class="font-bold pr-2">Added By:</p>
          <div class="flex">
            <p class="pr-2">Dylan Hrebenach</p>
            <Badge variant="primary-light">Admin</Badge>
          </div>
        </li>
        <li class="flex px-4">
          <p class="font-bold pr-2">Data Access Agreement:</p>
          <Badge variant="gray"
            ><a class="text-gray-600" href="data_access_agreement_signed_file.pdf"
              >data_access_agreement_signed_file.pdf</a
            ></Badge
          >
        </li>
        <li class="flex px-4 pb-4">
          <p class="font-bold pr-2">Uploaded On:</p>
          <p>2021-SEP-12 14:05</p>
        </li>
      </ul>
    </div>
  </div>
</main>
