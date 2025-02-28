import React from "react";

interface FeedbackCategory {
  value: string;
  label: string;
  description: string;
}

interface FeedbackModalProps {
  isOpen: boolean;
  selectedCategory: string;
  feedbackText: string;
  categories: FeedbackCategory[];
  onCategoryChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onFeedbackTextChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSubmit: () => void;
  onCancel: () => void;
}

const FeedbackModal: React.FC<FeedbackModalProps> = ({
  isOpen,
  selectedCategory,
  feedbackText,
  categories,
  onCategoryChange,
  onFeedbackTextChange,
  onSubmit,
  onCancel,
}) => {
  if (!isOpen) return null;

  return (
    <div className="modal modal-open">
      <div className="modal-box">
        <h3 className="font-bold text-lg mb-4">What could be improved?</h3>

        <div className="form-control w-full">
          <label className="label">
            <span className="label-text">Category</span>
          </label>
          <select
            className="select select-bordered w-full"
            value={selectedCategory}
            onChange={onCategoryChange}
            required
          >
            <option value="">Select a category</option>
            {categories.map((cat) => (
              <option key={cat.value} value={cat.value}>
                {cat.label}
              </option>
            ))}
          </select>
          {selectedCategory && (
            <label className="label">
              <span className="label-text-alt">
                {
                  categories.find((cat) => cat.value === selectedCategory)
                    ?.description
                }
              </span>
            </label>
          )}
        </div>

        <div className="form-control w-full mt-4">
          <label className="label">
            <span className="label-text">Additional Comments</span>
          </label>
          <textarea
            className="textarea textarea-bordered h-24"
            placeholder="Please provide more details about the issue..."
            value={feedbackText}
            onChange={onFeedbackTextChange}
          />
        </div>

        <div className="modal-action">
          <button
            className="btn btn-error"
            onClick={onSubmit}
            disabled={!selectedCategory}
          >
            Submit Feedback
          </button>
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default React.memo(FeedbackModal);
