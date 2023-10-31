import { test, expect } from "@playwright/test"

test("should display domain metadata, has a valid id", async ({ page }) => {
  await page.goto("/login")
  const metadataPanel = page.getByTestId("domain-metadata-panel")
  expect(await metadataPanel.isVisible()).toBe(true)

  // check if id is shown
  const badge = metadataPanel.getByTestId("badge")
  expect(await badge.textContent()).toBeTruthy()
})
