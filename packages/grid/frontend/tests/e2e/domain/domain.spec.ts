import { test, expect } from '@playwright/test';

test.describe('User Login', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('/datasets');
    // await expect(page.getByTestId('domain-logout')).toContainText(/logout/i);
  });

  test('should find config icon', async ({ page }) => {
    await page.goto('/datasets');
    await expect(page.getByTestId('domain-logout')).toContainText(/logout/i);
    await page.getByTestId('domain-config').click();
    await page.waitForURL('**/config**');
  });

  test('should find domain information', async ({ page }) => {
    await page.goto('/config');
    
    await expect(await page.getByTestId('tab-item-Domain')).toContainText(/Domain/i);
    await page.getByTestId('tab-item-Domain').click();
    await expect(page.getByTestId('profile-information')).toContainText(/Profile information/i);
    await expect(page.getByTestId('system-information')).toContainText(/System information/i);
    await expect(page.getByTestId('domain-deployed-on')).toContainText(/DEPLOYED ON/i);
  });

  test('should change domain information', async ({ page }) => {
    await page.goto('/config');
    
    await expect(await page.getByTestId('tab-item-Domain')).toContainText(/Domain/i);
    await page.getByTestId('tab-item-Domain').click();

    await test.step(`should change domain name`, async () => {
      await expect(await page.getByTestId('header-domain_name')).toBeVisible();
      await expect(await page.getByTestId('domain_name')).toBeVisible();
      await expect(await page.getByTestId('change-domain_name')).toContainText(/Change Domain Name/i);
      page.on('dialog', async dialog => {
        await page.getByTestId('change-domain_name').click();
        await page.getByTestId('domain-name').fill('Test Domain Name');
        await expect(await page.getByTestId('saveChange')).toContainText(/Save/i);
        await page.getByTestId('saveChange').click()
        await dialog.accept();
      });
      await page.waitForURL('**/config**');
      await expect(await page.getByTestId('domain_name')).toContainText(/Test Domain Name/i);
    });

    await test.step(`should change organization`, async () => {
      await expect(await page.getByTestId('header-organization')).toBeVisible();
      await expect(await page.getByTestId('organization')).toBeVisible();
      await expect(await page.getByTestId('change-organization')).toContainText(/Change Organization/i);
      page.on('dialog', async dialog => {
        await page.getByTestId('change-organization').click();
        await page.getByTestId('domain-organization').fill('Test Domain Organization');
        await expect(await page.getByTestId('saveChange')).toContainText(/Save/i);
        await page.getByTestId('saveChange').click()
        await dialog.accept();
      });
      await page.waitForURL('**/config**');
      await expect(await page.getByTestId('organization')).toContainText(/Test Domain Organization/i);
    });

    await test.step(`should change description`, async () => {
      await expect(await page.getByTestId('header-description')).toBeVisible();
      await expect(await page.getByTestId('description')).toBeVisible();
      await expect(await page.getByTestId('change-description')).toContainText(/Change Domain Name/i);
      page.on('dialog', async dialog => {
        await page.getByTestId('change-description').click();
        await page.getByTestId('domain-description').fill('Test Domain Description');
        await expect(await page.getByTestId('saveChange')).toContainText(/Save/i);
        await page.getByTestId('saveChange').click()
        await dialog.accept();
      });
      await page.waitForURL('**/config**');
      await expect(await page.getByTestId('description')).toContainText(/Test Domain Description/i);
    });

  });
});
