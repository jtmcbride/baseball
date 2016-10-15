class BabeRuthNumber < ActiveRecord::Base
  validates :team_id, uniqueness: true
  belongs_to :player
  belongs_to :team
end
