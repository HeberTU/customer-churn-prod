# -*- coding: utf-8 -*-
"""Schemas for bank data.

Created on: 05/03/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandera as pa
from pandera.typing import Series


class BankInputSchema(pa.SchemaModel):
    """Bank data input schema."""

    attrition_flag: Series[str] = pa.Field(coerce=True)
    customer_age: Series[int] = pa.Field(coerce=True, gt=0, lt=100)
    gender: Series[str] = pa.Field(coerce=True)
    dependent_count: Series[int] = pa.Field(coerce=True)
    marital_status: Series[str] = pa.Field(coerce=True)
    income_category: Series[str] = pa.Field(coerce=True)
    card_category: Series[str] = pa.Field(coerce=True)
    months_on_book: Series[int] = pa.Field(coerce=True)
    total_relationship_count: Series[int] = pa.Field(coerce=True, ge=0)
    months_inactive_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    contacts_count_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    credit_limit: Series[float] = pa.Field(coerce=True, ge=0)
    total_revolving_bal: Series[int] = pa.Field(coerce=True, ge=0)
    avg_open_to_buy: Series[float] = pa.Field(coerce=True, ge=0)
    total_amt_chng_q4_q1: Series[float] = pa.Field(coerce=True, ge=0)
    total_trans_amt: Series[int] = pa.Field(coerce=True, ge=0)
    total_trans_ct: Series[int] = pa.Field(coerce=True, ge=0)
    total_ct_chng_q4_q1: Series[float] = pa.Field(coerce=True, ge=0)
    avg_utilization_ratio: Series[float] = pa.Field(coerce=True, ge=0, le=1)

    @pa.check("attrition_flag", name="valid_attrition")
    def custom_check(cls, attrition_flag: Series[str]) -> Series[bool]:
        return attrition_flag.isin(
            ['Existing Customer', 'Attrited Customer'])

class BankOutputSchema(BankInputSchema):
    """Bank data output schema."""

    churn: Series[int] = pa.Field(coerce=True)

    @pa.check("churn", name="valid_churn")
    def custom_check(cls, churn: Series[int]) -> Series[bool]:
        return churn.isin([0, 1])