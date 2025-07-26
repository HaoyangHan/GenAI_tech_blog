---
title: "Markdown Mathematical Expression Guide"
description: "Guidelines for formatting mathematical expressions, calculations, and financial figures in blog posts"
date: "2024-03-21"
---

# Markdown Mathematical Expression Guide

This guide outlines the standards for formatting mathematical expressions, calculations, and financial figures in our blog posts. Following these guidelines ensures consistent rendering and readability across our technical content.

## 1. Inline Mathematical Expressions

### Basic Calculations
Use **code formatting** (backticks) for simple calculations:
```markdown
`800 × $300 = $240,000`
```

### Complex Mathematical Expressions
Use **LaTeX inline** format for complex mathematical expressions:
```markdown
\( \text{formula} \)
```

### ❌ Common Mistakes to Avoid
- Don't mix markdown bold (`**`) with mathematical operators
- Don't use asterisk (`*`) for multiplication in displayed text
- Don't leave currency symbols inconsistent

## 2. Block Mathematical Expressions

### Standard Formula Blocks
Use LaTeX block format with double dollar signs:
```markdown
$$
\text{Value Add} = (\text{Avg. Profit per Case}) \times (\text{Profit Gain per Insight}) \times (\text{Insights per Case})
$$
```

### Multi-line Equations
For equations that span multiple lines:
```markdown
$$
\begin{aligned}
\text{Total Cost} &= \text{Hours} \times \text{Rate} \\
&= 800 \times \$300 \\
&= \$240,000
\end{aligned}
$$
```

## 3. Currency and Number Formatting

### Currency Symbols
- Always include currency symbols: `$300/hour` not `300/hour`
- Place currency symbols before numbers: `$300` not `300$`
- For international currencies, use standard symbols: `€`, `£`, `¥`

### Large Numbers
- Use commas as thousand separators: `$1,000,000` not `$1000000`
- For very large numbers in tables, consider using:
  - Millions: `$156M` instead of `$156,000,000`
  - Billions: `$1.2B` instead of `$1,200,000,000`

### Percentages
- Use the percent symbol with no space: `65%` not `65 %`
- In calculations: `\( 65\% \)` for inline LaTeX

## 4. Mathematical Operators

### Multiplication
- Use `×` (HTML entity: `&times;`) in regular text
- Use `\times` in LaTeX expressions
- Use `×` in code-formatted calculations

### Division and Fractions
- Simple divisions: `cost/hour` or `cost per hour`
- Complex fractions in LaTeX: `\frac{numerator}{denominator}`

## 5. Mixed Content Formatting

### Calculations in Lists
```markdown
• **Total Cost:** `800 hours × $300/hour = $240,000`
• **Annual Savings:** `$156,000/memo × 1,000 memos/year = $156M`
```

### Table Formatting
```markdown
| Metric | Value | Calculation |
|:-------|:------|:------------|
| Hours | 800 | `200 × 4 weeks` |
| Cost | $240,000 | `800 × $300` |
```

### ⚠️ Critical Table Formatting Rules
- **NEVER put calculations inside table cells** - causes LaTeX parsing conflicts
- **NEVER use HTML spans inside table cells** - causes rendering conflicts  
- **NEVER use parentheses with calculations in cells** - triggers math renderer
- **Keep calculations outside tables** - use calculation details after the table
- **Use simple markdown formatting only** - bold (`**text**`) and code (`` `text` ``)
- **Move colored text outside tables** - use callouts or highlights after tables

### Example of What NOT To Do ❌
```markdown
| Cost | $240,000 | $84,000 (`280 × $300`) | -$156,000 |
```

### Example of What TO Do ✅  
```markdown
| Cost | $240,000 | $84,000 | -$156,000 |

**Calculation Details:**
- New cost: `280 × $300 = $84,000`
```

## 6. Best Practices

### 1. Consistency
- Use the same format for similar calculations throughout a document
- Maintain consistent precision in decimal places
- Keep currency formatting consistent

### 2. Readability
- Break long calculations into steps
- Use appropriate spacing around operators
- Align similar calculations in tables or lists

### 3. Technical Considerations
- Test LaTeX rendering in your markdown preview
- Ensure proper escaping of special characters
- Verify mobile responsiveness of formatted expressions

## Examples

### ✅ Correct Usage

```markdown
• **Investment Return:** `$100,000 × 1.15 = $115,000`
• **Annual Growth:** \( \frac{\$115,000 - \$100,000}{\$100,000} \times 100\% = 15\% \)
• **Total Value:** `$115,000 + ($115,000 × 0.05) = $120,750`
```

### ❌ Incorrect Usage

```markdown
• **Investment Return:** 100000 * 1.15 = **115000**
• **Annual Growth:** 115000-100000/100000 * 100 = **15%**
• **Total Value:** 115000 + (115000 * 0.05) = **120750**
```

## Related Resources

- [LaTeX Mathematical Expressions](https://katex.org/docs/supported.html)
- [GitHub Markdown Math Support](https://github.blog/2022-05-19-math-support-in-markdown/)
- [Unicode Mathematical Operators](https://www.unicode.org/charts/PDF/U2200.pdf) 