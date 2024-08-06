import { test, expect } from "@playwright/test"

test("should display datasite metadata, has a valid id", async ({ page }) => {
  await page.goto("/login")
  const metadataPanel = page.getByTestId("datasite-metadata-panel")
  expect(await metadataPanel.isVisible()).toBe(true)

  // check if id is shown
  const badge = metadataPanel.getByTestId("badge")
  expect(await badge.textContent()).toBeTruthy()
})
