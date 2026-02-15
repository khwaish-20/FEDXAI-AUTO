# 📊 Implementation Plan: Problem Section Layout Refactor

**Goal:** Modify the "Problem Statement", "Literature Review", and "Research Gap" sections in the presentation to be displayed as three equal-width horizontal columns.

## 1. Files to Modify
*   `pbl_ppt_template.html`

## 2. Proposed Changes

### Section: `#problem` (Grid Container)
*   **Current State:**
    *   Outer Grid: `grid md:grid-cols-5 gap-12`
    *   Problem Box: `md:col-span-3`
    *   Right Wrapper (Lit Review + Research Gap): `md:col-span-2 space-y-6`
*   **New State:**
    *   Outer Grid: `grid md:grid-cols-3 gap-8 items-stretch`
    *   Problem Box: Remove `md:col-span-3` and wrapper styling. Just a `div` with `glass p-8 rounded-3xl`.
    *   Lit Review Box: Remove wrapper `div`. Extract the inner `div` to be a direct sibling.
    *   Research Gap Box: Remove wrapper `div`. Extract the inner `div` to be a direct sibling.
    *   *Effect:* All three boxes will be direct children of a single `grid md:grid-cols-3` container, causing them to sit side-by-side with equal width and height.

## 3. Verification Plan
*   **Visual Check:** Open `pbl_ppt_template.html` in browser.
*   **Success Criteria:**
    1.  Three distinct boxes appear in a single row.
    2.  Each box takes up 1/3 of the width.
    3.  All boxes stretch to the same height.
