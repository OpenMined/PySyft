import { test, expect } from '@playwright/test';

test('has title', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/PyGrid/);
});

test('can login', async ({ page }) => {
  await page.goto('/login');

  await expect(page.getByTestId('deployed-on')).toContainText(/deployed on/i);

  await page.getByTestId('email').fill('info@openmined.org');
  await page.getByTestId('password').fill('changethis');

  await page.getByRole('button', { name: /login/i }).click();

  await page.waitForURL('**/datasets**');
});
