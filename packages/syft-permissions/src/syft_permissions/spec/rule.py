from pydantic import BaseModel, model_validator

from syft_permissions.spec.access import Access

UNSUPPORTED_TEMPLATES = ["{{.UserHash}}", "{{.Year}}", "{{.Month}}", "{{.Date}}"]


class Rule(BaseModel):
    pattern: str
    access: Access

    @model_validator(mode="after")
    def _reject_unsupported_templates(self) -> "Rule":
        for tpl in UNSUPPORTED_TEMPLATES:
            if tpl in self.pattern:
                raise ValueError(f"Unsupported template in pattern: {tpl}")
        return self
