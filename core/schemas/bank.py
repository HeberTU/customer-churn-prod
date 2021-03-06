# -*- coding: utf-8 -*-
"""Banck Schemas module.

Created on: 04/03/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""

from typing import Union, Optional
import pandera as pa
from pandera.typing import Series


class BankInputSchema(pa.SchemaModel):
    """Bank data input schema."""

    Attrition_Flag: Series[str] = pa.Field(coerce=True)
    Customer_Age: Series[int] = pa.Field(coerce=True, gt=0, lt=100)
    Gender: Series[str] = pa.Field(coerce=True)
    Dependent_count: Series[int] = pa.Field(coerce=True)
    Education_Level: Series[str] = pa.Field(coerce=True)
    Marital_Status: Series[str] = pa.Field(coerce=True)
    Income_Category: Series[str] = pa.Field(coerce=True)
    Card_Category: Series[str] = pa.Field(coerce=True)
    Months_on_book: Series[int] = pa.Field(coerce=True)
    Total_Relationship_Count: Series[int] = pa.Field(coerce=True, ge=0)
    Months_Inactive_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    Contacts_Count_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    Credit_Limit: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Revolving_Bal: Series[int] = pa.Field(coerce=True, ge=0)
    Avg_Open_To_Buy: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Amt_Chng_Q4_Q1: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Trans_Amt: Series[int] = pa.Field(coerce=True, ge=0)
    Total_Trans_Ct: Series[int] = pa.Field(coerce=True, ge=0)
    Total_Ct_Chng_Q4_Q1: Series[float] = pa.Field(coerce=True, ge=0)
    Avg_Utilization_Ratio: Series[float] = pa.Field(coerce=True, ge=0, le=1)

    @pa.check("Attrition_Flag", name="valid_attrition")
    def custom_check(cls, Attrition_Flag: Series[str]) -> Series[bool]:
        return Attrition_Flag.isin(
            ['Existing Customer', 'Attrited Customer'])


class BankOutputSchema(BankInputSchema):
    """Bank data output schema."""

    Churn: Series[int] = pa.Field(coerce=True)

    @pa.check("Churn", name="valid_churn")
    def custom_check(cls, Churn: Series[int]) -> Series[bool]:
        return Churn.isin([0, 1])


class BankMLSchema(pa.SchemaModel):
    """Bank data ml schema."""

    Customer_Age: Series[int] = pa.Field(coerce=True, gt=0, lt=100)
    Gender_Churn: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Dependent_count: Series[int] = pa.Field(coerce=True)
    Education_Level_Churn: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Marital_Status_Churn: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Income_Category_Churn: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Card_Category_Churn: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Months_on_book: Series[int] = pa.Field(coerce=True)
    Total_Relationship_Count: Series[int] = pa.Field(coerce=True, ge=0)
    Months_Inactive_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    Contacts_Count_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    Credit_Limit: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Revolving_Bal: Series[int] = pa.Field(coerce=True, ge=0)
    Avg_Open_To_Buy: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Amt_Chng_Q4_Q1: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Trans_Amt: Series[int] = pa.Field(coerce=True, ge=0)
    Total_Trans_Ct: Series[int] = pa.Field(coerce=True, ge=0)
    Total_Ct_Chng_Q4_Q1: Series[float] = pa.Field(coerce=True, ge=0)
    Avg_Utilization_Ratio: Series[float] = pa.Field(coerce=True, ge=0, le=1)


class BankMLSchemaInPlace(pa.SchemaModel):
    """Bank data ml inplace schema."""

    Customer_Age: Series[int] = pa.Field(coerce=True, gt=0, lt=100)
    Gender: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Dependent_count: Series[int] = pa.Field(coerce=True)
    Education_Level: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Marital_Status: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Income_Category: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Card_Category: Series[float] = pa.Field(coerce=True, ge=0, le=1)
    Months_on_book: Series[int] = pa.Field(coerce=True)
    Total_Relationship_Count: Series[int] = pa.Field(coerce=True, ge=0)
    Months_Inactive_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    Contacts_Count_12_mon: Series[int] = pa.Field(coerce=True, ge=0)
    Credit_Limit: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Revolving_Bal: Series[int] = pa.Field(coerce=True, ge=0)
    Avg_Open_To_Buy: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Amt_Chng_Q4_Q1: Series[float] = pa.Field(coerce=True, ge=0)
    Total_Trans_Amt: Series[int] = pa.Field(coerce=True, ge=0)
    Total_Trans_Ct: Series[int] = pa.Field(coerce=True, ge=0)
    Total_Ct_Chng_Q4_Q1: Series[float] = pa.Field(coerce=True, ge=0)
    Avg_Utilization_Ratio: Series[float] = pa.Field(coerce=True, ge=0, le=1)


def get_ml_schema(
        response: Optional[str]
) -> Union[BankMLSchema, BankMLSchemaInPlace]:
    """Factory function for ML Schema.

    input:
        response: string of response name [optional argument that could be
             used for naming variables or index y column]

    output:
            MLSchema.
    """

    ml_schemas = {
        'Churn': BankMLSchema
    }

    return ml_schemas.get(response, BankMLSchemaInPlace)
