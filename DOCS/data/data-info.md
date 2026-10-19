# The H&M mini dataset

This is a tiny slice (10 products, 5 customers, 10 purchases) of the Kaggle "H&M Personalized Fashion Recommendations" dataset. It lives in `data/H_and_M_mini_dataset/`, as both CSV (`csv-format/`) and parquet (partitioned folders). It has three related tables:

```
customers (5 rows)  ──< transactions (10 rows) >──  articles (10 rows)
  customer_id  PK         customer_id FK              article_id  PK
                          article_id  FK
```

## 1. `articles.csv`: the product catalog

One row per sellable item (a specific colour/variant). The columns are best read in groups:

| Group | Columns | Meaning |
|---|---|---|
| Identity | `article_id`, `product_code`, `prod_name` | `article_id` is the unique variant (10 digits). `product_code` is the first 7 digits, shared by all colour variants of one design. |
| Type | `product_type_name`, `product_group_name` | What it is (Blazer, Bikini top) and its broad group (Swimwear, Garment Upper body). |
| Look | `graphical_appearance_name`, `colour_group_name`, `perceived_colour_*` | Pattern (Solid, Denim) and colour. The perceived colour is how it looks to a shopper, such as "Dark" / "Black". |
| Org hierarchy | `department_name`, `index_name`, `index_group_name`, `section_name`, `garment_group_name` | H&M's internal merchandising tree, from broad (Ladieswear / Divided) to narrow (Womens Tailoring). |
| Text | `detail_desc` | Free-text marketing description. Good for embeddings and RAG. |

Every `*_no` / `*_code` column is just the numeric ID for the adjacent `*_name`. They are redundant, so you can ignore the IDs and use the names.

## 2. `customers.csv`: customer profiles

| Column | Meaning |
|---|---|
| `customer_id` | 64-char hashed ID (anonymized). |
| `FN` | 1.0 if they get fashion-news emails, else null. |
| `Active` | 1.0 if active for communication, else null. |
| `club_member_status` | ACTIVE / PRE-CREATE / LEFT CLUB. |
| `fashion_news_frequency` | NONE / Regularly / Monthly. |
| `age` | Age in years. |
| `postal_code` | Hashed postal code. |

Nulls in `FN` and `Active` mean "no", not missing data (customer `3d3f…` has both null and `NONE` news frequency).

## 3. `transactions.csv`: purchases (the glue table)

| Column | Meaning |
|---|---|
| `customer_id` → customers | Who bought. |
| `article_id` → articles | What they bought. |
| `t_dat` | Purchase date. |
| `price` | Normalized, not real currency (e.g. 0.0338). |
| `sales_channel_id` | 1 = store, 2 = online. |
| `year`, `month` | Partition columns for the parquet layout. |

## What stands out in this sample

- **One customer dominates.** `d65380…` made 9 of the 10 purchases, and the other 4 customers have at most 1 (here, 1 for `f43c9c…`), so 3 customers have no purchases. Real data is sparse like this too.
- **Mostly black items.** 7 of 10 articles are Black, and 3 are in swimwear. That makes it a good small test for "recommend something similar".
- **Missing description.** The Ellen Thong (`0507883014`) has an empty `detail_desc`. Handle nulls before embedding.
- **A repeated 2020-04-05 haul.** Several items bought the same day suggests a basket or order, though there is no explicit `order_id`.
- **Transactions are online only** (`sales_channel_id = 2`).

## How it relates to this project

`scripts/neon-db-creation/init_neon.py` loads these three CSVs into Neon: `articles` becomes `product_catalog`, `transactions` become `purchase_history` and (grouped by customer and date) `orders` / `order_items`, and `customers` become `customers` and `personalization`. Ids are made readable (`cust001`.., `prod001`..); order status is derived as delivered because the transactions are historical. The mapping rules are in `DOCS/data/user-item/schema-overview.md`. The `detail_desc` column is the natural field for semantic search.
