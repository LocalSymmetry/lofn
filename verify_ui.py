from playwright.sync_api import sync_playwright

def verify(page):
    print("Navigating...")
    page.goto("http://localhost:8000/")
    print("Waiting for selector...")
    page.wait_for_selector('text=LOFN /// ENGINE')
    print("Taking screenshot...")
    page.screenshot(path="verification.png")
    print("Done.")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            verify(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
