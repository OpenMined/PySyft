import get from 'lodash.get'

import permissionStrings from '@/i18n/en/permissions.json'
import commonStrings from '@/i18n/en/common.json'
import sectionsStrings from '@/i18n/en/sections.json'
import loginStrings from '@/i18n/en/login.json'
import usersStrings from '@/i18n/en/users.json'
import accountStrings from '@/i18n/en/account.json'
import pathStrings from '@/i18n/en/paths.json'
import settingsStrings from '@/i18n/en/settings.json'

function t(key: string, i18n_key = 'common') {
  const i18ns = {
    permissions: permissionStrings,
    common: commonStrings,
    sections: sectionsStrings,
    login: loginStrings,
    users: usersStrings,
    account: accountStrings,
    paths: pathStrings,
    settings: settingsStrings,
  }
  return get(i18ns[i18n_key], key?.toLowerCase())
}

export { t }
