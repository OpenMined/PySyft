import { test, expect } from '@playwright/test';

test.describe('User Login', () => {
  let page;

  test.beforeEach(async ({ page, baseURL }) => {
    await page.goto('/login');
    await expect(page.getByTestId('deployed-on')).toContainText(/deployed on/i);
  });

  test('should login an existing user', async ({ page }) => {
    await page.getByTestId('email').fill('dylan@openmined.org');
    await page.getByTestId('password').fill('dylan');

    await page.getByRole('button', { name: /login/i }).click();

    await page.waitForURL('**/datasets**');
  });
});
